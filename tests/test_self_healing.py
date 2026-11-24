import pytest

from controller.self_healing import (
    ActionKind,
    AuditEntry,
    GuardrailState,
    K8sActuatorAdapter,
    K8sPodRef,
    PDBBlockedError,
    PodTelemetry,
    SelfHealingEngine,
    SimulatedActuatorAdapter,
    TelemetryWindow,
    compute_z_score,
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

    # Live run drains at most one node at a time, and releases once drain completes
    guardrails = GuardrailState(action_cooldown_seconds=60.0)
    live_plan = engine.reconcile(two_bad_nodes, guardrails)
    assert actuator.drained_nodes == ["node-a"]
    assert AuditEntry("skip", ActionKind.CORDON_AND_DRAIN, "node-b", "max_one_drain") in (
        live_plan.audit_log
    )
    engine.complete_drain(guardrails, "node-a")
    next_plan = engine.reconcile(
        TelemetryWindow(timestamp=110.0, pods=two_bad_nodes.pods),
        guardrails,
    )
    assert (ActionKind.CORDON_AND_DRAIN, "node-b") in [
        (a.kind, a.target) for a in next_plan.actions
    ]

    # Failing pod restart twice triggers human escalation via ClusterActuator.page_human
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
    assert actuator.paged_targets == [("vllm-8b-bad", "restart_pod_failed_2_times")]
    assert AuditEntry("skip", ActionKind.MARK_POOL_DEGRADED, "pool-8b", "cooldown") in (
        second_try.audit_log
    )


def test_ttft_zscore_anomaly_with_elevated_kv_marks_pool_degraded() -> None:
    pool_health = PoolHealthStore()
    actuator = SimulatedActuatorAdapter()
    engine = SelfHealingEngine(actuator=actuator, pool_health=pool_health)

    plan = engine.reconcile(
        TelemetryWindow(
            timestamp=300.0,
            pods=[
                PodTelemetry(
                    pod="vllm-8b-2",
                    node="gpu-node-3",
                    pool="pool-8b",
                    kv_cache_usage=0.84,
                    gpu_temp_c=70.0,
                    ttft_ms_series=[100.0, 102.0, 99.0, 101.0, 450.0],
                )
            ],
        ),
        GuardrailState(),
    )
    assert (ActionKind.MARK_POOL_DEGRADED, "pool-8b") in [
        (a.kind, a.target) for a in plan.actions
    ]
    assert pool_health.get_state("pool-8b").state == HealthState.DEGRADED


def test_mad_zscore_ignores_jitter_and_survives_prior_outliers() -> None:
    # Tiny sub-ms jitter on flat baseline must NOT false-trigger
    assert compute_z_score([100.0, 100.0, 100.0, 100.0, 100.2]) < 1.0

    # Prior spike in baseline must NOT mask a subsequent 460ms spike
    assert compute_z_score([100.0, 450.0, 102.0, 99.0, 101.0, 460.0]) > 3.5


def test_multivariate_isolation_forest_flags_joint_degradation() -> None:
    pool_health = PoolHealthStore()
    actuator = SimulatedActuatorAdapter()
    baseline = [
        (100.0, 0.30, 62.0),
        (105.0, 0.35, 63.0),
        (98.0, 0.28, 61.0),
        (102.0, 0.32, 62.5),
        (104.0, 0.31, 63.0),
        (101.0, 0.33, 62.0),
        (99.0, 0.29, 61.5),
        (103.0, 0.34, 62.8),
        (100.0, 0.30, 62.2),
        (102.0, 0.31, 62.4),
    ]
    engine = SelfHealingEngine(
        actuator=actuator,
        pool_health=pool_health,
        baseline_features=baseline,
    )

    # Jointly elevated TTFT (280ms), KV (0.88), temp (81C) — below hard thresholds (95%, 85C)
    plan = engine.reconcile(
        TelemetryWindow(
            timestamp=400.0,
            pods=[
                PodTelemetry(
                    pod="vllm-8b-3",
                    node="gpu-node-4",
                    pool="pool-8b",
                    kv_cache_usage=0.88,
                    gpu_temp_c=81.0,
                    ttft_ms_series=[100.0, 102.0, 101.0, 280.0],
                )
            ],
        ),
        GuardrailState(),
    )
    assert (ActionKind.MARK_POOL_DEGRADED, "pool-8b") in [
        (a.kind, a.target) for a in plan.actions
    ]


def test_k8s_actuator_cordons_skips_daemonsets_and_retries_pdb_429() -> None:
    cordoned: list[str] = []
    evicted: list[str] = []
    attempts = {"vllm-0": 0}

    class _FakeK8sClient:
        def cordon_node(self, node: str) -> None:
            cordoned.append(node)

        def list_node_pods(self, node: str) -> list[K8sPodRef]:
            _ = node
            return [
                K8sPodRef(namespace="telemetry", name="dcgm-exporter", is_daemonset=True),
                K8sPodRef(namespace="llm", name="vllm-0", is_daemonset=False),
            ]

        def evict_pod(self, namespace: str, name: str) -> None:
            _ = namespace
            attempts[name] = attempts.get(name, 0) + 1
            if attempts[name] == 1:
                raise PDBBlockedError("429 Too Many Requests: PDB active")
            evicted.append(name)

        def delete_pod(self, namespace: str, name: str) -> None:
            _ = (namespace, name)

        def page_human(self, target: str, reason: str) -> None:
            _ = (target, reason)

    adapter = K8sActuatorAdapter(
        client=_FakeK8sClient(),
        max_eviction_retries=2,
        backoff_seconds=0.0,
    )
    adapter.cordon_and_drain("gpu-node-1", "xid_error")
    assert cordoned == ["gpu-node-1"]
    assert evicted == ["vllm-0"]

    # Exceeding max_eviction_retries raises RuntimeError
    attempts["vllm-0"] = 0
    adapter_strict = K8sActuatorAdapter(
        client=_FakeK8sClient(),
        max_eviction_retries=1,
        backoff_seconds=0.0,
    )
    with pytest.raises(RuntimeError, match="PDB"):
        adapter_strict.cordon_and_drain("gpu-node-1", "xid_error")
