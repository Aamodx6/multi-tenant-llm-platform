from __future__ import annotations

from datetime import UTC, datetime

import fakeredis
import pytest
from sqlalchemy import create_engine, text

from gateway.request_control import UsageEvent
from gateway.usage_rollup import (
    BillingRollupWorker,
    RedisStreamUsageSink,
    normalize_database_url,
)


def test_redis_stream_usage_sink_and_rollup_idempotent_dedup_and_xack() -> None:
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    db_engine = create_engine("sqlite:///:memory:")

    sink = RedisStreamUsageSink(redis_client=redis_client)
    ts_ms = int(datetime(2026, 11, 24, 14, 25, tzinfo=UTC).timestamp() * 1000)

    ev1 = UsageEvent(
        trace_id="trace-1",
        tenant_id="tenant-a",
        pool="pool-8b",
        tokens=150,
        prompt_tokens=100,
        completion_tokens=50,
        lora_name="tenant-a-sql",
        recorded_at_ms=ts_ms,
    )
    ev1_dup = UsageEvent(
        trace_id="trace-1",
        tenant_id="tenant-a",
        pool="pool-8b",
        tokens=150,
        prompt_tokens=100,
        completion_tokens=50,
        lora_name="tenant-a-sql",
        recorded_at_ms=ts_ms,
    )
    ev2 = UsageEvent(
        trace_id="trace-2",
        tenant_id="tenant-a",
        pool="pool-8b",
        tokens=50,
        prompt_tokens=30,
        completion_tokens=20,
        lora_name="tenant-a-sql",
        recorded_at_ms=ts_ms + 60_000,
    )

    sink.emit(ev1)
    sink.emit(ev1_dup)
    sink.emit(ev2)

    worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=db_engine,
        consumer_name="worker-1",
    )
    worker.ensure_schema()
    processed = worker.run_once()
    assert processed == 2  # unique trace_ids inserted; duplicate trace-1 skipped by ON CONFLICT

    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT tenant_id, pool, lora_name, bucket_hour, "
                "prompt_tokens, completion_tokens, total_tokens, request_count "
                "FROM tenant_usage_hourly"
            )
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "tenant-a"
    assert row[1] == "pool-8b"
    assert row[2] == "tenant-a-sql"
    assert str(row[3]).startswith("2026-11-24 14:00:00")
    assert row[4] == 130
    assert row[5] == 70
    assert row[6] == 200
    assert row[7] == 2

    pending = redis_client.xpending(sink.stream_key, worker.group_name)
    assert pending["pending"] == 0


def test_billing_rollup_worker_xautoclaim_recovers_stale_pending_and_skips_xack_on_failure() -> (
    None
):
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    db_engine = create_engine("sqlite:///:memory:")
    sink = RedisStreamUsageSink(redis_client=redis_client)
    ts_ms = int(datetime(2026, 11, 24, 15, 5, tzinfo=UTC).timestamp() * 1000)

    sink.emit(
        UsageEvent(
            trace_id="trace-crash-1",
            tenant_id="tenant-b",
            pool="pool-8b",
            tokens=90,
            prompt_tokens=60,
            completion_tokens=30,
            recorded_at_ms=ts_ms,
        )
    )

    class _BrokenEngine:
        def begin(self) -> object:
            raise RuntimeError("postgres transient outage")

    crashed_worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=_BrokenEngine(),  # type: ignore[arg-type]
        consumer_name="dead-worker",
    )
    crashed_worker._ensure_group()
    with pytest.raises(RuntimeError, match="postgres transient outage"):
        crashed_worker.run_once()

    # Must remain unacknowledged in PEL because commit never succeeded
    assert redis_client.xpending(sink.stream_key, crashed_worker.group_name)["pending"] == 1

    healthy_worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=db_engine,
        consumer_name="recovery-worker",
    )
    healthy_worker.ensure_schema()
    recovered = healthy_worker.recover_stale(min_idle_ms=0)
    assert recovered == 1
    assert redis_client.xpending(sink.stream_key, healthy_worker.group_name)["pending"] == 0

    with db_engine.connect() as conn:
        total = conn.execute(
            text("SELECT total_tokens, request_count FROM tenant_usage_hourly")
        ).fetchone()
    assert total == (90, 1)


def test_caller_reused_trace_id_cannot_bypass_billing_and_trimmed_entries_do_not_crash() -> None:
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    db_engine = create_engine("sqlite:///:memory:")
    sink = RedisStreamUsageSink(redis_client=redis_client)

    # Same caller trace_id, distinct server-generated event_id per completion
    msg_a = UsageEvent(
        trace_id="attacker-trace-id",
        event_id="evt-1",
        tenant_id="tenant-a",
        pool="pool-8b",
        tokens=40,
        prompt_tokens=20,
        completion_tokens=20,
    )
    msg_b = UsageEvent(
        trace_id="attacker-trace-id",
        event_id="evt-2",
        tenant_id="tenant-a",
        pool="pool-8b",
        tokens=60,
        prompt_tokens=30,
        completion_tokens=30,
    )
    sink.emit(msg_a)
    sink.emit(msg_b)

    worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=db_engine,
        consumer_name="worker-a",
    )
    worker.ensure_schema()
    assert worker.run_once() == 2

    # Simulated trimmed entry (msg_id, None) from XAUTOCLAIM must be ignored + XACKed cleanly
    trimmed = worker._process_entries([("9999999999999-0", None)])
    assert trimmed == 0

    with db_engine.connect() as conn:
        total = conn.execute(
            text("SELECT total_tokens, request_count FROM tenant_usage_hourly")
        ).fetchone()
    assert total == (100, 2)


def test_normalize_database_url_and_poison_pill_dlq_recovery() -> None:
    assert normalize_database_url("postgresql://u:p@db:5432/llm") == (
        "postgresql+psycopg://u:p@db:5432/llm"
    )
    assert normalize_database_url("postgres://u:p@db:5432/llm") == (
        "postgresql+psycopg://u:p@db:5432/llm"
    )
    assert normalize_database_url("sqlite:///:memory:") == "sqlite:///:memory:"

    redis_client = fakeredis.FakeRedis(decode_responses=True)
    db_engine = create_engine("sqlite:///:memory:")
    sink = RedisStreamUsageSink(redis_client=redis_client)

    # First event triggers a deterministic DB failure (poison pill); second is valid
    sink.emit(
        UsageEvent(
            trace_id="poison-1",
            event_id="evt-poison",
            tenant_id="tenant-a",
            pool="pool-8b",
            tokens=10,
            prompt_tokens=5,
            completion_tokens=5,
        )
    )
    sink.emit(
        UsageEvent(
            trace_id="good-1",
            event_id="evt-good",
            tenant_id="tenant-a",
            pool="pool-8b",
            tokens=40,
            prompt_tokens=20,
            completion_tokens=20,
        )
    )

    class _PoisonOnceEngine:
        def begin(self) -> object:
            raise RuntimeError("transient failure")

    bad_worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=_PoisonOnceEngine(),  # type: ignore[arg-type]
        consumer_name="dead-w",
        max_deliveries=2,
    )
    bad_worker._ensure_group()
    with pytest.raises(RuntimeError):
        bad_worker.run_once(count=1)

    # Recovery worker claims stale message (delivery #2 >= max_deliveries=2),
    # quarantines poison pill to DLQ, and processes good-1
    good_worker = BillingRollupWorker(
        redis_client=redis_client,
        db_engine=db_engine,
        consumer_name="good-w",
        max_deliveries=2,
    )
    good_worker.ensure_schema()
    good_worker.recover_stale(min_idle_ms=0, count=10)
    good_worker.run_once(count=10)

    dlq_entries = redis_client.xrange(f"{sink.stream_key}:dlq")
    assert dlq_entries is not None
    assert len(dlq_entries) == 1
    dlq_fields = dlq_entries[0][1]
    assert dlq_fields is not None
    assert dlq_fields["event_id"] == "evt-poison"
    assert redis_client.xpending(sink.stream_key, good_worker.group_name)["pending"] == 0

    with db_engine.connect() as conn:
        total = conn.execute(
            text("SELECT total_tokens, request_count FROM tenant_usage_hourly")
        ).fetchone()
    assert total == (40, 1)
