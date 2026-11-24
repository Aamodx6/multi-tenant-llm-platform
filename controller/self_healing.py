from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from gateway.pool_health import PoolHealth


class ActionKind(StrEnum):
    """Permitted slow-path remediation actions."""

    CORDON_AND_DRAIN = "cordon_and_drain"
    MARK_POOL_DEGRADED = "mark_pool_degraded"
    RESTART_POD = "restart_pod"
    ESCALATE_TO_HUMAN = "escalate_to_human"


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
    last_action_at: dict[tuple[ActionKind, str], float] = field(default_factory=dict)
    failure_counts: dict[tuple[ActionKind, str], int] = field(default_factory=dict)
    action_cooldown_seconds: float = 60.0
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
    """Production ClusterActuator adapter delegating node drains, restarts, and paging."""

    def __init__(
        self,
        *,
        drain_fn: Callable[[str, str], None],
        restart_fn: Callable[[str, str], None],
        page_fn: Callable[[str, str], None],
    ) -> None:
        self._drain_fn = drain_fn
        self._restart_fn = restart_fn
        self._page_fn = page_fn

    def cordon_and_drain(self, node: str, reason: str) -> None:
        self._drain_fn(node, reason)

    def restart_pod(self, pod: str, reason: str) -> None:
        self._restart_fn(pod, reason)

    def page_human(self, target: str, reason: str) -> None:
        self._page_fn(target, reason)


def compute_z_score(series: list[float]) -> float:
    """Compute the latest-sample z-score against prior points in a telemetry series."""
    if len(series) < 3:
        return 0.0
    baseline = series[:-1]
    mean = sum(baseline) / len(baseline)
    variance = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    std = math.sqrt(variance)
    if std < 1e-6:
        return 0.0 if abs(series[-1] - mean) < 1e-6 else 10.0
    return (series[-1] - mean) / std


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
        z_score_threshold: float = 3.0,
    ) -> None:
        self._actuator = actuator
        self._pool_health = pool_health
        self._degraded_ttl = degraded_ttl_seconds
        self._thermal_threshold_c = thermal_threshold_c
        self._kv_threshold = kv_cache_threshold
        self._z_score_threshold = z_score_threshold

    def complete_drain(self, guardrails: GuardrailState, node: str) -> None:
        """Clear an in-progress node drain once eviction completes."""
        guardrails.draining_nodes.discard(node)

    def reconcile(
        self,
        window: TelemetryWindow,
        guardrails: GuardrailState,
    ) -> RemediationPlan:
        """Score a TelemetryWindow, plan actions under guardrails, and invoke ClusterActuator."""
        scores: dict[str, float] = {}
        candidates: list[tuple[ActionKind, str, str]] = []

        for pod in window.pods:
            z = compute_z_score(pod.ttft_ms_series)
            scores[pod.pod] = z
            has_hw_fault = (
                pod.xid_errors > 0
                or pod.ecc_errors > 0
                or pod.gpu_temp_c >= self._thermal_threshold_c
            )
            if has_hw_fault:
                candidates.append((ActionKind.CORDON_AND_DRAIN, pod.node, "gpu_hardware_fault"))
            if pod.kv_cache_usage >= self._kv_threshold or pod.oom_restarts > 0:
                candidates.append((ActionKind.MARK_POOL_DEGRADED, pod.pool, "kv_or_oom_pressure"))
                candidates.append((ActionKind.RESTART_POD, pod.pod, "kv_or_oom_pressure"))
            elif z >= self._z_score_threshold and pod.kv_cache_usage >= 0.80:
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
                actions.append(
                    PlannedAction(kind=kind, target=target, reason=reason, executed=True)
                )
                audit_log.append(AuditEntry("executed", kind, target, reason))
            except RuntimeError as exc:
                fails = guardrails.failure_counts.get(key, 0) + 1
                guardrails.failure_counts[key] = fails
                audit_log.append(AuditEntry("failed", kind, target, f"attempt_{fails}:{exc}"))
                if fails >= guardrails.max_failures_before_page:
                    esc_reason = f"{kind}_failed_{fails}_times"
                    self._actuator.page_human(target, esc_reason)
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
