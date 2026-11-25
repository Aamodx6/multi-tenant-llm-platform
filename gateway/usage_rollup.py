from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
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

from gateway.db import normalize_database_url
from gateway.request_control import UsageEvent

_LOGGER = logging.getLogger(__name__)
DEFAULT_STREAM_KEY = "usage:events"
DEFAULT_GROUP_NAME = "billing-rollup"
__all__ = [
    "BillingRollupWorker",
    "DEFAULT_GROUP_NAME",
    "DEFAULT_STREAM_KEY",
    "RedisStreamUsageSink",
    "TENANT_USAGE_HOURLY",
    "USAGE_EVENT_DEDUP",
    "normalize_database_url",
]

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


def _decode_stream_id(raw_id: Any) -> str:
    return raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)


def _decode_stream_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
        )
        for k, v in fields.items()
    }


def _event_to_fields(event: UsageEvent) -> dict[str, str]:
    prompt_tokens = event.prompt_tokens
    completion_tokens = event.completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0 and event.tokens > 0:
        completion_tokens = event.tokens
    recorded_at_ms = event.recorded_at_ms or int(time.time() * 1000)
    return {
        "event_id": event.event_id or event.trace_id,
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
        self._lock = threading.Lock()
        self.dropped_events_count = 0

    def emit(self, event: UsageEvent) -> None:
        """Append a UsageEvent to Redis Streams, buffering briefly on transient errors."""
        fields = _event_to_fields(event)
        try:
            with self._lock:
                has_pending = bool(self._pending_buffer)
            if has_pending:
                self.flush(limit=64)
            self._redis.xadd(
                self.stream_key,
                fields,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
        except redis.RedisError:
            with self._lock:
                if (
                    self._pending_buffer.maxlen is not None
                    and len(self._pending_buffer) >= self._pending_buffer.maxlen
                ):
                    self.dropped_events_count += 1
                    _LOGGER.warning(
                        "Usage buffer full (%d); evicted oldest buffered event (dropped=%d).",
                        self._pending_buffer.maxlen,
                        self.dropped_events_count,
                    )
                self._pending_buffer.append(fields)

    def flush(self, *, limit: int | None = None) -> int:
        """Flush buffered fallback events to Redis Streams in one pipelined round-trip."""
        with self._lock:
            pending = len(self._pending_buffer)
            if not pending:
                return 0
            count = pending if limit is None else min(limit, pending)
            batch = [self._pending_buffer[i] for i in range(count)]
        pipe = self._redis.pipeline(transaction=False)
        for fields in batch:
            pipe.xadd(
                self.stream_key,
                fields,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
        pipe.execute()
        with self._lock:
            for _ in range(min(count, len(self._pending_buffer))):
                self._pending_buffer.popleft()
        return count


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
        max_deliveries: int = 5,
    ) -> None:
        self._redis = redis_client
        self._db_engine = db_engine
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name
        self._max_deliveries = max(1, max_deliveries)

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

    def _quarantine_poison_pills(
        self, claimed: Sequence[tuple[str, Mapping[str, Any] | None]]
    ) -> list[tuple[str, Mapping[str, Any] | None]]:
        if not claimed:
            return []
        healthy: list[tuple[str, Mapping[str, Any] | None]] = []
        dlq_key = f"{self.stream_key}:dlq"
        for raw_id, fields in claimed:
            msg_id = _decode_stream_id(raw_id)
            deliveries = 0
            try:
                pending_info = self._redis.xpending_range(
                    self.stream_key,
                    self.group_name,
                    min=msg_id,
                    max=msg_id,
                    count=1,
                )
                if pending_info:
                    deliveries = int(pending_info[0].get("times_delivered", 0))
            except redis.RedisError:
                deliveries = 0

            if deliveries >= self._max_deliveries:
                dlq_fields: dict[str, str] = {"original_id": msg_id, "deliveries": str(deliveries)}
                if fields:
                    dlq_fields.update(_decode_stream_fields(fields))
                self._redis.xadd(dlq_key, dlq_fields, maxlen=100_000, approximate=True)  # type: ignore[arg-type]
                self._redis.xack(self.stream_key, self.group_name, msg_id)
                _LOGGER.warning(
                    "Quarantined poison-pill usage event %s after %d deliveries to %s",
                    msg_id,
                    deliveries,
                    dlq_key,
                )
            else:
                healthy.append((raw_id, fields))
        return healthy

    def recover_stale(self, *, min_idle_ms: int = 60_000, count: int = 100) -> int:
        """Reclaim abandoned pending entries via XAUTOCLAIM cursor, DLQ poison pills, and XACK."""
        self._ensure_group()
        total_processed = 0
        cursor = "0-0"
        while True:
            result = self._redis.xautoclaim(
                name=self.stream_key,
                groupname=self.group_name,
                consumername=self.consumer_name,
                min_idle_time=min_idle_ms,
                start_id=cursor,
                count=count,
            )
            if not result:
                break
            next_cursor = result[0]
            claimed: list[tuple[str, Mapping[str, Any] | None]] = result[1] or []
            if claimed:
                healthy = self._quarantine_poison_pills(claimed)
                if healthy:
                    total_processed += self._process_entries(healthy)
            cursor_str = (
                next_cursor.decode("utf-8") if isinstance(next_cursor, bytes) else str(next_cursor)
            )
            if not claimed or cursor_str == "0-0" or cursor_str == cursor:
                break
            cursor = cursor_str
        return total_processed

    def _flatten_xreadgroup(self, response: Any) -> list[tuple[str, Mapping[str, Any]]]:
        if not response:
            return []
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for _stream, stream_entries in response:
            entries.extend(stream_entries)
        return entries

    def _process_entries(self, entries: Sequence[tuple[str, Mapping[str, Any] | None]]) -> int:
        message_ids: list[str] = []
        parsed_events: list[dict[str, Any]] = []
        now = datetime.now(tz=UTC)

        for raw_id, fields in entries:
            msg_id = _decode_stream_id(raw_id)
            message_ids.append(msg_id)
            if fields is None:
                continue
            decoded = _decode_stream_fields(fields)
            try:
                dedup_id = decoded.get("event_id") or decoded["trace_id"]
                ts_ms = int(decoded.get("recorded_at_ms", "0") or 0)
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC) if ts_ms > 0 else now
                bucket_hour = dt.replace(minute=0, second=0, microsecond=0)
                parsed_events.append(
                    {
                        "trace_id": dedup_id,
                        "tenant_id": decoded["tenant_id"],
                        "pool": decoded["pool"],
                        "lora_name": decoded.get("lora_name", ""),
                        "bucket_hour": bucket_hour,
                        "prompt_tokens": int(decoded.get("prompt_tokens", "0")),
                        "completion_tokens": int(decoded.get("completion_tokens", "0")),
                        "total_tokens": int(decoded.get("total_tokens", "0")),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue

        if not parsed_events:
            if message_ids:
                self._redis.xack(self.stream_key, self.group_name, *message_ids)
            return 0

        dedup_count = 0
        with self._db_engine.begin() as conn:
            ordered = sorted(parsed_events, key=lambda row: str(row["trace_id"]))
            # Deduplicate within the batch first so multi-row VALUES inserts cleanly
            unique_by_id: dict[str, dict[str, Any]] = {}
            for item in ordered:
                unique_by_id.setdefault(str(item["trace_id"]), item)
            unique_items = list(unique_by_id.values())

            placeholders = ", ".join(
                f"(:tid_{idx}, :created_at)" for idx in range(len(unique_items))
            )
            params: dict[str, Any] = {"created_at": now}
            for idx, item in enumerate(unique_items):
                params[f"tid_{idx}"] = item["trace_id"]
            dedup_stmt = text(
                f"INSERT INTO usage_event_dedup (trace_id, created_at) "
                f"VALUES {placeholders} "
                "ON CONFLICT (trace_id) DO NOTHING RETURNING trace_id"
            ).bindparams(bindparam("created_at", type_=DateTime(timezone=True)))
            inserted_ids = {
                str(row[0]) for row in conn.execute(dedup_stmt, params).fetchall()
            }
            accepted = [item for item in unique_items if str(item["trace_id"]) in inserted_ids]

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
                for (tenant_id, pool, lora_name, bucket_hour), totals in sorted(agg.items()):
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
    database_url = normalize_database_url(os.environ["DATABASE_URL"])
    consumer = os.environ.get("HOSTNAME", "billing-worker-0")

    redis_client = redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=5.0,
    )
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
