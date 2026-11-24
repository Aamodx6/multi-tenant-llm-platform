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
