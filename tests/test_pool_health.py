import fakeredis

from gateway.pool_health import HealthState, PoolHealthStore, RedisPoolHealthAdapter


def test_pool_health_defaults_to_healthy_and_honors_degraded_ttl_and_hysteresis() -> None:
    clock = [100.0]
    store = PoolHealthStore(
        recovery_cooldown_seconds=30.0,
        min_healthy_probes=2,
        now_fn=lambda: clock[0],
    )

    assert store.get_state("pool-8b").state == HealthState.HEALTHY

    # Slow-path remediation marks pool degraded for 20s
    store.set_degraded("pool-8b", reason="kv_cache_high", ttl_seconds=20.0)
    status = store.get_state("pool-8b")
    assert status.state == HealthState.DEGRADED
    assert status.reason == "kv_cache_high"

    # Fast-path healthy probe during TTL cannot clear degraded override
    clock[0] = 110.0
    store.record_probe("pool-8b", ok=True)
    assert store.get_state("pool-8b").state == HealthState.DEGRADED

    # After TTL expires without probe failure, override auto-clears
    clock[0] = 125.0
    assert store.get_state("pool-8b").state == HealthState.HEALTHY

    # Probe-induced degradation requires cooldown + min_healthy_probes consecutive successes
    store.record_probe("pool-8b", ok=False)
    assert store.get_state("pool-8b").state == HealthState.DEGRADED
    clock[0] = 160.0
    store.record_probe("pool-8b", ok=True)
    assert store.get_state("pool-8b").state == HealthState.DEGRADED

    store.record_probe("pool-8b", ok=True)
    assert store.get_state("pool-8b").state == HealthState.HEALTHY


def test_probe_failures_escalate_to_unhealthy_and_reset_streak() -> None:
    clock = [10.0]
    store = PoolHealthStore(
        recovery_cooldown_seconds=10.0,
        min_healthy_probes=2,
        unhealthy_failure_threshold=3,
        now_fn=lambda: clock[0],
    )

    assert store.record_probe("pool-a", ok=False).state == HealthState.DEGRADED
    assert store.record_probe("pool-a", ok=False).state == HealthState.DEGRADED
    assert store.record_probe("pool-a", ok=False).state == HealthState.UNHEALTHY

    # set_degraded must not downgrade an already UNHEALTHY pool
    degraded_call = store.set_degraded("pool-a", "kv_pressure", ttl_seconds=10.0)
    assert degraded_call.state == HealthState.UNHEALTHY

    clock[0] = 30.0
    assert store.record_probe("pool-a", ok=True).state == HealthState.UNHEALTHY
    assert store.record_probe("pool-a", ok=False).state == HealthState.UNHEALTHY
    clock[0] = 50.0
    assert store.record_probe("pool-a", ok=True).state == HealthState.UNHEALTHY
    assert store.record_probe("pool-a", ok=True).state == HealthState.HEALTHY


def test_redis_pool_health_shares_state_across_pods_and_expires_ttl() -> None:
    clock = [100.0]
    r = fakeredis.FakeRedis(decode_responses=True)
    controller_health = RedisPoolHealthAdapter(
        redis_client=r,
        recovery_cooldown_seconds=10.0,
        min_healthy_probes=2,
        now_fn=lambda: clock[0],
    )
    gateway_health = RedisPoolHealthAdapter(
        redis_client=r,
        recovery_cooldown_seconds=10.0,
        min_healthy_probes=2,
        now_fn=lambda: clock[0],
    )

    controller_health.set_degraded("pool-8b", reason="kv_pressure", ttl_seconds=20.0)
    assert gateway_health.get_state("pool-8b").state == HealthState.DEGRADED
    assert gateway_health.get_state("pool-8b").reason == "kv_pressure"

    # After TTL expires without probe failures, get_state unblocks without needing active probe
    clock[0] = 125.0
    assert gateway_health.get_state("pool-8b").state == HealthState.HEALTHY

    # Probe failures still require consecutive healthy probes to recover
    gateway_health.record_probe("pool-8b", ok=False)
    assert controller_health.get_state("pool-8b").state == HealthState.DEGRADED
    clock[0] = 140.0
    assert gateway_health.record_probe("pool-8b", ok=True).state == HealthState.DEGRADED
    assert controller_health.record_probe("pool-8b", ok=True).state == HealthState.HEALTHY
