from controller.self_healing import (
    ActionKind,
    GuardrailState,
    PodTelemetry,
    SelfHealingEngine,
    SimulatedActuatorAdapter,
    TelemetryWindow,
)
from gateway.pool_health import HealthState, PoolHealthStore


def test_reconcile_drains_xid_node_and_marks_kv_saturated_pool_degraded() -> None:
    pool_health = PoolHealthStore()
    actuator = SimulatedActuatorAdapter()
    engine = SelfHealingEngine(actuator=actuator, pool_health=pool_health)

    window = TelemetryWindow(
        timestamp=1000.0,
        pods=[
            PodTelemetry(
                pod="vllm-8b-0",
                node="gpu-node-1",
                pool="pool-8b",
                kv_cache_usage=0.55,
                gpu_temp_c=68.0,
                xid_errors=2,
                ecc_errors=0,
                oom_restarts=0,
                ttft_ms_series=[120.0, 125.0, 118.0, 122.0, 121.0],
            ),
            PodTelemetry(
                pod="vllm-8b-1",
                node="gpu-node-2",
                pool="pool-8b",
                kv_cache_usage=0.97,
                gpu_temp_c=72.0,
                xid_errors=0,
                ecc_errors=0,
                oom_restarts=1,
                ttft_ms_series=[120.0, 124.0, 119.0, 121.0, 480.0],
            ),
        ],
    )

    plan = engine.reconcile(window, GuardrailState())

    kinds = [(a.kind, a.target) for a in plan.actions]
    assert (ActionKind.CORDON_AND_DRAIN, "gpu-node-1") in kinds
    assert (ActionKind.MARK_POOL_DEGRADED, "pool-8b") in kinds
    assert (ActionKind.RESTART_POD, "vllm-8b-1") in kinds
    assert plan.anomaly_scores["vllm-8b-1"] > 1.5

    # Actuator executed slow-path mutations; PoolHealth shifted without mutating Deployment args
    assert actuator.drained_nodes == ["gpu-node-1"]
    assert actuator.restarted_pods == ["vllm-8b-1"]
    assert pool_health.get_state("pool-8b").state == HealthState.DEGRADED


def test_guardrails_dry_run_max_one_drain_cooldown_and_two_failure_escalation() -> None:
    pool_health = PoolHealthStore()
    actuator = SimulatedActuatorAdapter(failing_targets={"vllm-8b-bad"})
    engine = SelfHealingEngine(actuator=actuator, pool_health=pool_health)

    two_bad_nodes = TelemetryWindow(
        timestamp=100.0,
        pods=[
            PodTelemetry(
                pod="p1",
                node="node-a",
                pool="pool-8b",
                kv_cache_usage=0.4,
                gpu_temp_c=90.0,
            ),
            PodTelemetry(
                pod="p2",
                node="node-b",
                pool="pool-8b",
                kv_cache_usage=0.4,
                gpu_temp_c=91.0,
            ),
        ],
    )

    # Dry-run records planned actions without mutating cluster
    dry_plan = engine.reconcile(two_bad_nodes, GuardrailState(dry_run=True))
    assert len(dry_plan.actions) == 2
    assert all(not a.executed for a in dry_plan.actions)
    assert actuator.drained_nodes == []

    # Live run drains at most one node at a time
    guardrails = GuardrailState(action_cooldown_seconds=60.0)
    live_plan = engine.reconcile(two_bad_nodes, guardrails)
    assert actuator.drained_nodes == ["node-a"]
    assert any("max_one_drain" in entry for entry in live_plan.audit_log)

    # Failing pod restart twice triggers human escalation
    failing_pod_window = TelemetryWindow(
        timestamp=200.0,
        pods=[
            PodTelemetry(
                pod="vllm-8b-bad",
                node="node-c",
                pool="pool-8b",
                kv_cache_usage=0.98,
                gpu_temp_c=70.0,
            )
        ],
    )
    first_try = engine.reconcile(failing_pod_window, guardrails)
    assert all(a.kind != ActionKind.ESCALATE_TO_HUMAN for a in first_try.actions)

    # Second failure (MARK_POOL_DEGRADED in cooldown; failed pod restart retries & escalates)
    second_try = engine.reconcile(
        TelemetryWindow(timestamp=210.0, pods=failing_pod_window.pods),
        guardrails,
    )
    assert (ActionKind.ESCALATE_TO_HUMAN, "vllm-8b-bad") in [
        (a.kind, a.target) for a in second_try.actions
    ]
    assert any("skip:mark_pool_degraded:pool-8b:cooldown" in e for e in second_try.audit_log)
