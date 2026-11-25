import fakeredis
import pytest
import redis

from gateway.tenant_budget import (
    AuthenticationError,
    BudgetExceededError,
    InMemoryBudgetAdapter,
    PriorityClass,
    RedisPostgresAdapter,
    TenantBudget,
    TenantRecord,
    hash_api_key,
)


def test_authorize_reserve_and_reconcile_refunds_unused_tokens() -> None:
    clock = [1000.0]
    tenant = TenantRecord(
        tenant_id="acme",
        priority=PriorityClass.PAID,
        tpm_limit=100,
        max_concurrency=4,
    )
    adapter = InMemoryBudgetAdapter(
        keys_by_hash={hash_api_key("sk-acme-1"): tenant},
        now_fn=lambda: clock[0],
    )
    budget = TenantBudget(adapter=adapter)

    with pytest.raises(AuthenticationError):
        budget.authorize_and_reserve("sk-unknown", "pool-8b", est_tokens=10)

    # Reserve 80 tokens out of 100 TPM
    permit = budget.authorize_and_reserve("sk-acme-1", "pool-8b", est_tokens=80)
    assert permit.tenant_id == "acme"
    assert permit.priority == PriorityClass.PAID
    assert permit.reserved_tokens == 80

    # Remaining is 20 -> requesting 30 must fail
    with pytest.raises(BudgetExceededError):
        budget.authorize_and_reserve("sk-acme-1", "pool-8b", est_tokens=30)

    # Reconcile actual usage of 30 -> 50 refunded -> 70 available now
    budget.reconcile(permit, actual_tokens=30)
    second = budget.authorize_and_reserve("sk-acme-1", "pool-8b", est_tokens=60)
    assert second.reserved_tokens == 60


def test_redis_adapter_atomic_lua_bucket_and_conservative_degraded_fallback() -> None:
    clock = [500.0]
    tenant = TenantRecord(
        tenant_id="beta",
        priority=PriorityClass.FREE,
        tpm_limit=100,
        max_concurrency=2,
    )
    key_hash = hash_api_key("sk-beta-1")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    adapter = RedisPostgresAdapter(
        redis_client=fake_redis,
        pg_lookup=lambda h: tenant if h == key_hash else None,
        degraded_tpm_ratio=0.25,
        now_fn=lambda: clock[0],
    )
    budget = TenantBudget(adapter=adapter)

    p1 = budget.authorize_and_reserve("sk-beta-1", "pool-8b", est_tokens=70)
    assert p1.degraded_mode is False
    with pytest.raises(BudgetExceededError):
        budget.authorize_and_reserve("sk-beta-1", "pool-8b", est_tokens=40)

    budget.reconcile(p1, actual_tokens=20)
    p2 = budget.authorize_and_reserve("sk-beta-1", "pool-8b", est_tokens=40)
    assert p2.reserved_tokens == 40

    # Simulate Redis outage: degrades transparently to 25% conservative local TPM (25 tokens)
    class _BrokenRedis:
        def get(self, *_args: object, **_kwargs: object) -> None:
            raise redis.ConnectionError("redis down")

        def set(self, *_args: object, **_kwargs: object) -> None:
            raise redis.ConnectionError("redis down")

        def eval(self, *_args: object, **_kwargs: object) -> object:
            raise redis.ConnectionError("redis down")

    adapter_down = RedisPostgresAdapter(
        redis_client=_BrokenRedis(),  # type: ignore[arg-type]
        pg_lookup=lambda h: tenant if h == key_hash else None,
        degraded_tpm_ratio=0.25,
        now_fn=lambda: clock[0],
    )
    degraded_budget = TenantBudget(adapter=adapter_down)
    p3 = degraded_budget.authorize_and_reserve("sk-beta-1", "pool-8b", est_tokens=20)
    assert p3.degraded_mode is True
    with pytest.raises(BudgetExceededError):
        degraded_budget.authorize_and_reserve("sk-beta-1", "pool-8b", est_tokens=10)


def test_concurrency_cap_blocks_excess_inflight_until_reconcile() -> None:
    tenant = TenantRecord(
        tenant_id="gamma",
        priority=PriorityClass.FREE,
        tpm_limit=1000,
        max_concurrency=1,
    )
    budget = TenantBudget(
        adapter=InMemoryBudgetAdapter({hash_api_key("sk-gamma"): tenant})
    )
    permit = budget.authorize_and_reserve("sk-gamma", "pool-8b", est_tokens=10)
    with pytest.raises(BudgetExceededError, match="concurrency"):
        budget.authorize_and_reserve("sk-gamma", "pool-8b", est_tokens=10)

    budget.reconcile(permit, actual_tokens=5)
    next_permit = budget.authorize_and_reserve("sk-gamma", "pool-8b", est_tokens=10)
    assert next_permit.tenant_id == "gamma"


def test_redis_adapter_shares_concurrency_across_pods_sets_reject_ttl_and_scales_degraded() -> None:
    tenant = TenantRecord(
        tenant_id="delta",
        priority=PriorityClass.PAID,
        tpm_limit=1000,
        max_concurrency=1,
    )
    kh = hash_api_key("sk-delta")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    pod1 = TenantBudget(
        adapter=RedisPostgresAdapter(
            redis_client=fake_redis,
            pg_lookup=lambda h: tenant if h == kh else None,
        )
    )
    pod2 = TenantBudget(
        adapter=RedisPostgresAdapter(
            redis_client=fake_redis,
            pg_lookup=lambda h: tenant if h == kh else None,
        )
    )

    # Rejected over-cap reserve still sets EXPIRE on the Redis bucket key
    with pytest.raises(BudgetExceededError):
        pod1.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=1500)
    assert fake_redis.ttl("tpm:delta") > 0

    # Pod 1 holds the single concurrency slot -> Pod 2 is blocked until Pod 1 reconciles
    p1 = pod1.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=10)
    with pytest.raises(BudgetExceededError, match="concurrency"):
        pod2.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=10)
    pod1.reconcile(p1, actual_tokens=10)
    p2 = pod2.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=10)
    pod2.reconcile(p2, actual_tokens=10)

    # Redis outage scales degraded budget by gateway_replicas (1000 * 0.40 / 4 = 100 TPM/pod)
    class _BrokenRedis:
        def get(self, *_args: object, **_kwargs: object) -> None:
            raise redis.ConnectionError("down")

        def set(self, *_args: object, **_kwargs: object) -> None:
            raise redis.ConnectionError("down")

        def eval(self, *_args: object, **_kwargs: object) -> object:
            raise redis.ConnectionError("down")

    degraded_pod = TenantBudget(
        adapter=RedisPostgresAdapter(
            redis_client=_BrokenRedis(),  # type: ignore[arg-type]
            pg_lookup=lambda h: tenant if h == kh else None,
            degraded_tpm_ratio=0.40,
            gateway_replicas=4,
        )
    )
    p3 = degraded_pod.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=80)
    assert p3.degraded_mode is True
    with pytest.raises(BudgetExceededError):
        degraded_pod.authorize_and_reserve("sk-delta", "pool-8b", est_tokens=30)


def test_budget_preserves_negative_debt_on_overrun_in_memory_and_redis() -> None:
    clock = [1000.0]
    tenant = TenantRecord("overrun", PriorityClass.PAID, tpm_limit=60, max_concurrency=4)
    kh = hash_api_key("sk-overrun")

    mem_budget = TenantBudget(
        adapter=InMemoryBudgetAdapter({kh: tenant}, now_fn=lambda: clock[0])
    )
    p = mem_budget.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=50)
    # Actual usage was 110 (50 over full 60 TPM cap -> balance should be -50)
    mem_budget.reconcile(p, actual_tokens=110)

    # After 30s (refill = 30 tokens -> balance = -20), even a 1-token request must fail
    clock[0] = 1030.0
    with pytest.raises(BudgetExceededError) as exc_info:
        mem_budget.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=1)
    assert exc_info.value.tenant_id == "overrun"
    assert exc_info.value.priority == PriorityClass.PAID

    # After 60s total (refill = 60 -> balance = +10), a 5-token request succeeds
    clock[0] = 1060.0
    assert (
        mem_budget.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=5).reserved_tokens
        == 5
    )

    # Same invariant in RedisPostgresAdapter, including across cache-hit pods
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    r_clock = [2000.0]
    pod_a = TenantBudget(
        adapter=RedisPostgresAdapter(
            redis_client=fake_redis,
            pg_lookup=lambda h: tenant if h == kh else None,
            now_fn=lambda: r_clock[0],
        )
    )
    pod_b = TenantBudget(
        adapter=RedisPostgresAdapter(
            redis_client=fake_redis,
            pg_lookup=lambda _h: None,  # Forces Redis tenant_key cache hit
            now_fn=lambda: r_clock[0],
        )
    )
    pa = pod_a.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=50)
    pod_b.reconcile(pa, actual_tokens=110)
    r_clock[0] = 2030.0
    with pytest.raises(BudgetExceededError):
        pod_b.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=1)
    r_clock[0] = 2060.0
    pb = pod_b.authorize_and_reserve("sk-overrun", "pool-8b", est_tokens=5)
    pod_b.reconcile(pb, actual_tokens=5)


def test_redis_reconcile_retries_transient_error_instead_of_leaking_concurrency() -> None:
    tenant = TenantRecord("retry-co", PriorityClass.PAID, tpm_limit=1000, max_concurrency=1)
    kh = hash_api_key("sk-retry")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    fail_reconcile = {"active": False}

    class _FlakyRedis:
        def __init__(self, inner: fakeredis.FakeRedis) -> None:
            self._inner = inner

        def get(self, *args: object, **kwargs: object) -> object:
            return self._inner.get(*args, **kwargs)  # type: ignore[arg-type]

        def set(self, *args: object, **kwargs: object) -> object:
            return self._inner.set(*args, **kwargs)  # type: ignore[arg-type]

        def eval(self, script: str, numkeys: int, *keys_and_args: str | int | float) -> object:
            if fail_reconcile["active"] and "ZREM" in script:
                raise redis.ConnectionError("transient drop")
            return self._inner.eval(script, numkeys, *keys_and_args)

    adapter = RedisPostgresAdapter(
        redis_client=_FlakyRedis(fake_redis),  # type: ignore[arg-type]
        pg_lookup=lambda h: tenant if h == kh else None,
    )
    assert adapter._permit_ttl >= 600.0
    budget = TenantBudget(adapter=adapter)

    p1 = budget.authorize_and_reserve("sk-retry", "pool-8b", est_tokens=10)
    fail_reconcile["active"] = True
    budget.reconcile(p1, actual_tokens=10)  # Queues failed reconcile instead of dropping slot
    fail_reconcile["active"] = False

    # Next reserve drains queued reconcile first, freeing concurrency slot
    p2 = budget.authorize_and_reserve("sk-retry", "pool-8b", est_tokens=10)
    budget.reconcile(p2, actual_tokens=10)
