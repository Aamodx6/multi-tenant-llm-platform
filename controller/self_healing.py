from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from sklearn.ensemble import IsolationForest

from gateway.pool_health import PoolHealth
from observability.metrics import MetricsRegistry


class ActionKind(StrEnum):
    """Permitted slow-path remediation actions."""

    CORDON_AND_DRAIN = "cordon_and_drain"
    MARK_POOL_DEGRADED = "mark_pool_degraded"
    RESTART_POD = "restart_pod"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class PDBBlockedError(RuntimeError):
    """Raised by K8sNodeClient.evict_pod when a PodDisruptionBudget returns HTTP 429."""


@dataclass(frozen=True)
class K8sPodRef:
    """Reference to a pod scheduled on a Kubernetes node."""

    namespace: str
    name: str
    is_daemonset: bool = False


class K8sNodeClient(Protocol):
    """Port abstracting Kubernetes cordon, list, policy/v1 eviction, delete, and paging."""

    def cordon_node(self, node: str) -> None: ...

    def list_node_pods(self, node: str) -> list[K8sPodRef]: ...

    def evict_pod(self, namespace: str, name: str) -> None: ...

    def delete_pod(self, namespace: str, name: str) -> None: ...

    def page_human(self, target: str, reason: str) -> None: ...


@dataclass(frozen=True)
class TelemetryFeature:
    """Multivariate feature vector used for Isolation Forest baseline fitting."""

    ttft_ms: float
    kv_cache_usage: float
    gpu_temp_c: float

    def to_row(self) -> list[float]:
        return [self.ttft_ms, self.kv_cache_usage, self.gpu_temp_c]


@dataclass(frozen=True)
class PodTelemetry:
    """Single-pod GPU and inference telemetry sample."""

    pod: str
    node: str
    pool: str
    kv_cache_usage: float
    gpu_temp_c: float
    xid_errors: int = 0
    ecc_errors: int = 0
    oom_restarts: int = 0
    ttft_ms_series: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TelemetryWindow:
    """Time-ordered snapshot of pod telemetry evaluated by SelfHealingEngine."""

    timestamp: float
    pods: list[PodTelemetry]


@dataclass
class GuardrailState:
    """Mutable guardrail state across reconciliation cycles."""

    dry_run: bool = False
    draining_nodes: set[str] = field(default_factory=set)
    draining_started_at: dict[str, float] = field(default_factory=dict)
    last_action_at: dict[tuple[ActionKind, str], float] = field(default_factory=dict)
    last_escalated_at: dict[tuple[ActionKind, str], float] = field(default_factory=dict)
    failure_counts: dict[tuple[ActionKind, str], int] = field(default_factory=dict)
    action_cooldown_seconds: float = 60.0
    drain_timeout_seconds: float = 300.0
    escalation_cooldown_seconds: float = 300.0
    max_failures_before_page: int = 2


@dataclass(frozen=True)
class PlannedAction:
    """Single remediation step emitted in a RemediationPlan."""

    kind: ActionKind
    target: str
    reason: str
    executed: bool = False


@dataclass(frozen=True)
class AuditEntry:
    """Structured audit log entry emitted by SelfHealingEngine."""

    outcome: str
    kind: ActionKind
    target: str
    detail: str


@dataclass(frozen=True)
class RemediationPlan:
    """Deterministic output of one SelfHealingEngine reconciliation pass."""

    anomaly_scores: dict[str, float]
    actions: list[PlannedAction]
    audit_log: list[AuditEntry]


class ClusterActuator(Protocol):
    """Seam between SelfHealingEngine and Kubernetes node/pod mutations."""

    def cordon_and_drain(self, node: str, reason: str) -> None: ...

    def restart_pod(self, pod: str, reason: str) -> None: ...

    def page_human(self, target: str, reason: str) -> None: ...


class SimulatedActuatorAdapter:
    """In-memory ClusterActuator adapter for deterministic controller and chaos tests."""

    def __init__(self, *, failing_targets: set[str] | None = None) -> None:
        self.drained_nodes: list[str] = []
        self.restarted_pods: list[str] = []
        self.paged_targets: list[tuple[str, str]] = []
        self.failing_targets = set(failing_targets or set())

    def cordon_and_drain(self, node: str, reason: str) -> None:
        if node in self.failing_targets:
            raise RuntimeError(f"Simulated drain failure on {node}: {reason}")
        self.drained_nodes.append(node)

    def restart_pod(self, pod: str, reason: str) -> None:
        if pod in self.failing_targets:
            raise RuntimeError(f"Simulated restart failure on {pod}: {reason}")
        self.restarted_pods.append(pod)

    def page_human(self, target: str, reason: str) -> None:
        self.paged_targets.append((target, reason))


class K8sActuatorAdapter:
    """Production ClusterActuator performing cordon and policy/v1 PDB-aware Eviction."""

    def __init__(
        self,
        *,
        client: K8sNodeClient,
        default_namespace: str = "llm",
        max_eviction_retries: int = 5,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._client = client
        self._default_namespace = default_namespace
        self._max_eviction_retries = max_eviction_retries
        self._backoff_seconds = backoff_seconds

    def cordon_and_drain(self, node: str, reason: str) -> None:
        """Cordon node and evict non-DaemonSet pods with 429 PDB retry."""
        _ = reason
        self._client.cordon_node(node)
        for pod in self._client.list_node_pods(node):
            if pod.is_daemonset:
                continue
            self._evict_with_retry(pod)

    def _evict_with_retry(self, pod: K8sPodRef) -> None:
        for attempt in range(self._max_eviction_retries):
            try:
                self._client.evict_pod(pod.namespace, pod.name)
                return
            except PDBBlockedError as exc:
                if attempt + 1 >= self._max_eviction_retries:
                    raise RuntimeError(
                        f"PDB blocked eviction of {pod.namespace}/{pod.name} "
                        f"after {self._max_eviction_retries} attempts"
                    ) from exc
                if self._backoff_seconds > 0.0:
                    time.sleep(self._backoff_seconds * (2**attempt))

    def restart_pod(self, pod: str, reason: str) -> None:
        _ = reason
        namespace, _, name = pod.partition("/")
        if not name:
            namespace, name = self._default_namespace, pod
        self._client.delete_pod(namespace, name)

    def page_human(self, target: str, reason: str) -> None:
        self._client.page_human(target, reason)


def compute_z_score(series: list[float], *, min_delta: float = 25.0) -> float:
    """Compute Iglewicz-Hoaglin MAD Modified Z-Score for upward latency spikes."""
    if len(series) < 3:
        return 0.0
    baseline = series[:-1]
    med = statistics.median(baseline)
    deviations = [abs(x - med) for x in baseline]
    mad = statistics.median(deviations)
    signed_diff = series[-1] - med
    if mad < 1e-6:
        return 0.0 if signed_diff < min_delta else 10.0
    if signed_diff < min_delta:
        return 0.0
    return 0.6745 * signed_diff / mad


class SelfHealingEngine:
    """Deep module combining anomaly scoring with guardrail-checked remediation planning."""

    def __init__(
        self,
        *,
        actuator: ClusterActuator,
        pool_health: PoolHealth,
        degraded_ttl_seconds: float = 120.0,
        thermal_threshold_c: float = 85.0,
        kv_cache_threshold: float = 0.95,
        z_score_threshold: float = 3.5,
        baseline_features: Sequence[TelemetryFeature] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._actuator = actuator
        self._pool_health = pool_health
        self._degraded_ttl = degraded_ttl_seconds
        self._thermal_threshold_c = thermal_threshold_c
        self._kv_threshold = kv_cache_threshold
        self._z_score_threshold = z_score_threshold
        self._metrics = metrics
        self._iso_forest: IsolationForest | None = None
        if baseline_features and len(baseline_features) >= 5:
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit([f.to_row() for f in baseline_features])
            self._iso_forest = model

    def complete_drain(self, guardrails: GuardrailState, node: str) -> None:
        """Clear an in-progress node drain once eviction completes."""
        guardrails.draining_nodes.discard(node)
        guardrails.draining_started_at.pop(node, None)

    def _is_multivariate_anomaly(self, pod: PodTelemetry) -> bool:
        if self._iso_forest is None or not pod.ttft_ms_series:
            return False
        feature = TelemetryFeature(
            ttft_ms=pod.ttft_ms_series[-1],
            kv_cache_usage=pod.kv_cache_usage,
            gpu_temp_c=pod.gpu_temp_c,
        )
        return int(self._iso_forest.predict([feature.to_row()])[0]) == -1

    def reconcile(
        self,
        window: TelemetryWindow,
        guardrails: GuardrailState,
    ) -> RemediationPlan:
        """Score a TelemetryWindow, plan actions under guardrails, and invoke ClusterActuator."""
        for node in list(guardrails.draining_nodes):
            started = guardrails.draining_started_at.get(node)
            if (
                started is not None
                and (window.timestamp - started) >= guardrails.drain_timeout_seconds
            ):
                self.complete_drain(guardrails, node)

        scores: dict[str, float] = {}
        candidates: list[tuple[ActionKind, str, str]] = []

        for pod in window.pods:
            z = compute_z_score(pod.ttft_ms_series)
            scores[pod.pod] = z
            if self._metrics is not None:
                self._metrics.anomaly_score.labels(pod=pod.pod, pool=pod.pool).set(z)
            has_hw_fault = (
                pod.xid_errors > 0
                or pod.ecc_errors > 0
                or pod.gpu_temp_c >= self._thermal_threshold_c
            )
            if has_hw_fault:
                candidates.append((ActionKind.CORDON_AND_DRAIN, pod.node, "gpu_hardware_fault"))
            if pod.kv_cache_usage >= self._kv_threshold or pod.oom_restarts > 0:
                candidates.append((ActionKind.MARK_POOL_DEGRADED, pod.pool, "kv_or_oom_pressure"))
                if pod.oom_restarts > 0 or pod.kv_cache_usage >= 0.98:
                    candidates.append((ActionKind.RESTART_POD, pod.pod, "kv_or_oom_pressure"))
            elif (z >= self._z_score_threshold and pod.kv_cache_usage >= 0.80) or (
                self._is_multivariate_anomaly(pod)
            ):
                candidates.append(
                    (ActionKind.MARK_POOL_DEGRADED, pod.pool, "ttft_anomaly_pressure")
                )

        actions: list[PlannedAction] = []
        audit_log: list[AuditEntry] = []
        seen: set[tuple[ActionKind, str]] = set()
        for kind, target, reason in candidates:
            key = (kind, target)
            if key in seen:
                continue
            seen.add(key)

            last_ts = guardrails.last_action_at.get(key)
            in_cooldown = (
                last_ts is not None
                and (window.timestamp - last_ts) < guardrails.action_cooldown_seconds
            )
            if in_cooldown:
                audit_log.append(AuditEntry("skip", kind, target, "cooldown"))
                continue

            if (
                kind == ActionKind.CORDON_AND_DRAIN
                and guardrails.draining_nodes
                and target not in guardrails.draining_nodes
            ):
                audit_log.append(AuditEntry("skip", kind, target, "max_one_drain"))
                continue

            if guardrails.dry_run:
                actions.append(
                    PlannedAction(kind=kind, target=target, reason=reason, executed=False)
                )
                audit_log.append(AuditEntry("dry_run", kind, target, reason))
                continue

            try:
                self._execute_action(kind, target, reason)
                guardrails.last_action_at[key] = window.timestamp
                guardrails.failure_counts[key] = 0
                if kind == ActionKind.CORDON_AND_DRAIN:
                    guardrails.draining_nodes.add(target)
                    guardrails.draining_started_at[target] = window.timestamp
                actions.append(
                    PlannedAction(kind=kind, target=target, reason=reason, executed=True)
                )
                audit_log.append(AuditEntry("executed", kind, target, reason))
                if self._metrics is not None:
                    self._metrics.remediation_actions_total.labels(
                        kind=str(kind), outcome="executed"
                    ).inc()
            except RuntimeError as exc:
                fails = guardrails.failure_counts.get(key, 0) + 1
                guardrails.failure_counts[key] = fails
                audit_log.append(AuditEntry("failed", kind, target, f"attempt_{fails}:{exc}"))
                if self._metrics is not None:
                    self._metrics.remediation_actions_total.labels(
                        kind=str(kind), outcome="failed"
                    ).inc()
                last_esc = guardrails.last_escalated_at.get(key)
                esc_ready = (
                    last_esc is None
                    or (window.timestamp - last_esc) >= guardrails.escalation_cooldown_seconds
                )
                if fails >= guardrails.max_failures_before_page and esc_ready:
                    esc_reason = f"{kind}_failed_{fails}_times"
                    self._actuator.page_human(target, esc_reason)
                    guardrails.last_escalated_at[key] = window.timestamp
                    actions.append(
                        PlannedAction(
                            kind=ActionKind.ESCALATE_TO_HUMAN,
                            target=target,
                            reason=esc_reason,
                            executed=True,
                        )
                    )
                    audit_log.append(
                        AuditEntry("escalate", ActionKind.ESCALATE_TO_HUMAN, target, "human_page")
                    )

        return RemediationPlan(anomaly_scores=scores, actions=actions, audit_log=audit_log)

    def _execute_action(self, kind: ActionKind, target: str, reason: str) -> None:
        if kind == ActionKind.CORDON_AND_DRAIN:
            self._actuator.cordon_and_drain(target, reason)
        elif kind == ActionKind.RESTART_POD:
            self._actuator.restart_pod(target, reason)
        elif kind == ActionKind.MARK_POOL_DEGRADED:
            self._pool_health.set_degraded(target, reason, ttl_seconds=self._degraded_ttl)
