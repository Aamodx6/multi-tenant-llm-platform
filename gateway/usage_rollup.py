from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import redis
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    bindparam,
    create_engine,
    text,
)

from gateway.request_control import UsageEvent

DEFAULT_STREAM_KEY = "usage:events"
DEFAULT_GROUP_NAME = "billing-rollup"

_METADATA = MetaData()

USAGE_EVENT_DEDUP = Table(
    "usage_event_dedup",
    _METADATA,
    Column("trace_id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

TENANT_USAGE_HOURLY = Table(
    "tenant_usage_hourly",
    _METADATA,
    Column("tenant_id", Text, nullable=False),
    Column("pool", Text, nullable=False),
    Column("lora_name", Text, nullable=False, server_default=""),
    Column("bucket_hour", DateTime(timezone=True), nullable=False),
    Column("prompt_tokens", BigInteger, nullable=False, server_default="0"),
    Column("completion_tokens", BigInteger, nullable=False, server_default="0"),
    Column("total_tokens", BigInteger, nullable=False, server_default="0"),
    Column("request_count", BigInteger, nullable=False, server_default="0"),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("tenant_id", "pool", "lora_name", "bucket_hour"),
)


def _event_to_fields(event: UsageEvent) -> dict[str, str]:
    prompt_tokens = event.prompt_tokens
    completion_tokens = event.completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0 and event.tokens > 0:
        completion_tokens = event.tokens
    recorded_at_ms = event.recorded_at_ms or int(time.time() * 1000)
    return {
        "trace_id": event.trace_id,
        "tenant_id": event.tenant_id,
        "pool": event.pool,
        "lora_name": event.lora_name or "",
        "prompt_tokens": str(prompt_tokens),
        "completion_tokens": str(completion_tokens),
        "total_tokens": str(event.tokens),
        "recorded_at_ms": str(recorded_at_ms),
    }


class RedisStreamUsageSink:
    """Durable non-blocking usage sink publishing UsageEvents to a Redis Stream."""

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        stream_key: str = DEFAULT_STREAM_KEY,
        maxlen: int = 1_000_000,
        max_buffer: int = 4096,
    ) -> None:
        self._redis = redis_client
        self.stream_key = stream_key
        self._maxlen = maxlen
        self._pending_buffer: deque[dict[str, str]] = deque(maxlen=max_buffer)

    def emit(self, event: UsageEvent) -> None:
        """Append a UsageEvent to Redis Streams, buffering briefly on transient errors."""
        fields = _event_to_fields(event)
        try:
            self.flush()
            self._redis.xadd(
                self.stream_key,
                fields,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
        except redis.RedisError:
            self._pending_buffer.append(fields)

    def flush(self) -> int:
        """Flush any buffered fallback events to Redis Streams."""
        flushed = 0
        while self._pending_buffer:
            fields = self._pending_buffer[0]
            self._redis.xadd(
                self.stream_key,
                fields,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
            self._pending_buffer.popleft()
            flushed += 1
        return flushed


class BillingRollupWorker:
    """Redis Streams consumer persisting deduplicated hourly tenant usage into Postgres."""

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        db_engine: Engine,
        stream_key: str = DEFAULT_STREAM_KEY,
        group_name: str = DEFAULT_GROUP_NAME,
        consumer_name: str = "billing-worker-0",
    ) -> None:
        self._redis = redis_client
        self._db_engine = db_engine
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name

    def ensure_schema(self) -> None:
        """Create idempotency and hourly usage rollup tables if missing."""
        _METADATA.create_all(self._db_engine)
        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self._redis.xgroup_create(
                self.stream_key,
                self.group_name,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def run_once(self, *, count: int = 100, block_ms: int | None = None) -> int:
        """Read new entries via XREADGROUP, persist transactionally, then XACK."""
        self._ensure_group()
        response = self._redis.xreadgroup(
            groupname=self.group_name,
            consumername=self.consumer_name,
            streams={self.stream_key: ">"},
            count=count,
            block=block_ms,
        )
        entries = self._flatten_xreadgroup(response)
        if not entries:
            return 0
        return self._process_entries(entries)

    def recover_stale(self, *, min_idle_ms: int = 60_000, count: int = 100) -> int:
        """Reclaim abandoned pending entries via XAUTOCLAIM, persist, then XACK."""
        self._ensure_group()
        result = self._redis.xautoclaim(
            name=self.stream_key,
            groupname=self.group_name,
            consumername=self.consumer_name,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        claimed: list[tuple[str, Mapping[str, Any]]] = result[1] if result else []
        if not claimed:
            return 0
        return self._process_entries(claimed)

    def _flatten_xreadgroup(self, response: Any) -> list[tuple[str, Mapping[str, Any]]]:
        if not response:
            return []
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for _stream, stream_entries in response:
            entries.extend(stream_entries)
        return entries

    def _process_entries(self, entries: list[tuple[str, Mapping[str, Any]]]) -> int:
        message_ids: list[str] = []
        parsed_events: list[dict[str, Any]] = []
        now = datetime.now(tz=UTC)

        for raw_id, fields in entries:
            msg_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            message_ids.append(msg_id)
            decoded = {
                (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                )
                for k, v in fields.items()
            }
            ts_ms = int(decoded.get("recorded_at_ms", "0") or 0)
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC) if ts_ms > 0 else now
            bucket_hour = dt.replace(minute=0, second=0, microsecond=0)
            parsed_events.append(
                {
                    "trace_id": decoded["trace_id"],
                    "tenant_id": decoded["tenant_id"],
                    "pool": decoded["pool"],
                    "lora_name": decoded.get("lora_name", ""),
                    "bucket_hour": bucket_hour,
                    "prompt_tokens": int(decoded.get("prompt_tokens", "0")),
                    "completion_tokens": int(decoded.get("completion_tokens", "0")),
                    "total_tokens": int(decoded.get("total_tokens", "0")),
                }
            )

        dedup_count = 0
        with self._db_engine.begin() as conn:
            accepted: list[dict[str, Any]] = []
            dedup_stmt = text(
                "INSERT INTO usage_event_dedup (trace_id, created_at) "
                "VALUES (:trace_id, :created_at) "
                "ON CONFLICT (trace_id) DO NOTHING RETURNING trace_id"
            ).bindparams(bindparam("created_at", type_=DateTime(timezone=True)))
            for item in parsed_events:
                inserted = conn.execute(
                    dedup_stmt, {"trace_id": item["trace_id"], "created_at": now}
                ).fetchone()
                if inserted is not None:
                    accepted.append(item)

            dedup_count = len(accepted)
            if accepted:
                agg: dict[tuple[str, str, str, datetime], dict[str, int]] = defaultdict(
                    lambda: {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "request_count": 0,
                    }
                )
                for item in accepted:
                    key = (
                        str(item["tenant_id"]),
                        str(item["pool"]),
                        str(item["lora_name"]),
                        item["bucket_hour"],
                    )
                    bucket = agg[key]
                    bucket["prompt_tokens"] += int(item["prompt_tokens"])
                    bucket["completion_tokens"] += int(item["completion_tokens"])
                    bucket["total_tokens"] += int(item["total_tokens"])
                    bucket["request_count"] += 1

                rollup_stmt = text(
                    "INSERT INTO tenant_usage_hourly "
                    "(tenant_id, pool, lora_name, bucket_hour, "
                    "prompt_tokens, completion_tokens, total_tokens, request_count, updated_at) "
                    "VALUES (:tenant_id, :pool, :lora_name, :bucket_hour, "
                    ":prompt_tokens, :completion_tokens, :total_tokens, "
                    ":request_count, :updated_at) "
                    "ON CONFLICT (tenant_id, pool, lora_name, bucket_hour) DO UPDATE SET "
                    "prompt_tokens = tenant_usage_hourly.prompt_tokens + excluded.prompt_tokens, "
                    "completion_tokens = "
                    "tenant_usage_hourly.completion_tokens + excluded.completion_tokens, "
                    "total_tokens = tenant_usage_hourly.total_tokens + excluded.total_tokens, "
                    "request_count = tenant_usage_hourly.request_count + excluded.request_count, "
                    "updated_at = excluded.updated_at"
                ).bindparams(
                    bindparam("bucket_hour", type_=DateTime(timezone=True)),
                    bindparam("updated_at", type_=DateTime(timezone=True)),
                )
                for (tenant_id, pool, lora_name, bucket_hour), totals in agg.items():
                    conn.execute(
                        rollup_stmt,
                        {
                            "tenant_id": tenant_id,
                            "pool": pool,
                            "lora_name": lora_name,
                            "bucket_hour": bucket_hour,
                            "prompt_tokens": totals["prompt_tokens"],
                            "completion_tokens": totals["completion_tokens"],
                            "total_tokens": totals["total_tokens"],
                            "request_count": totals["request_count"],
                            "updated_at": now,
                        },
                    )

        # Acknowledge only after DB transaction commit succeeded.
        self._redis.xack(self.stream_key, self.group_name, *message_ids)
        return dedup_count


def main() -> int:
    """Run the Redis Streams -> Postgres hourly billing worker until interrupted."""
    redis_url = os.environ["REDIS_URL"]
    database_url = os.environ["DATABASE_URL"]
    consumer = os.environ.get("HOSTNAME", "billing-worker-0")

    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    db_engine = create_engine(database_url, pool_pre_ping=True)
    worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=db_engine,
        consumer_name=consumer,
    )
    worker.ensure_schema()
    while True:
        worker.recover_stale()
        worker.run_once(block_ms=2000)


if __name__ == "__main__":
    raise SystemExit(main())
