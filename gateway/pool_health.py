from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class HealthState(StrEnum):
    """Operational state of a model pool."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class PoolStatus:
    """Snapshot of a model pool's operational state."""

    pool: str
    state: HealthState
    reason: str | None = None
    override_until: float = 0.0


class PoolHealth(Protocol):
    """Seam between fast-path RequestControl and slow-path SelfHealingEngine."""

    def get_state(self, pool: str) -> PoolStatus: ...

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus: ...

    def record_probe(self, pool: str, *, ok: bool) -> PoolStatus: ...


@dataclass
class _PoolRecord:
    state: HealthState = HealthState.HEALTHY
    reason: str | None = None
    override_until: float = 0.0
    last_transition_at: float = 0.0
    consecutive_ok: int = 0
    consecutive_fail: int = 0


class PoolHealthStore:
    """In-memory PoolHealth store with TTL overrides and anti-flap hysteresis."""

    def __init__(
        self,
        *,
        recovery_cooldown_seconds: float = 30.0,
        min_healthy_probes: int = 2,
        unhealthy_failure_threshold: int = 3,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recovery_cooldown = recovery_cooldown_seconds
        self._min_healthy_probes = min_healthy_probes
        self._unhealthy_threshold = unhealthy_failure_threshold
        self._now = now_fn
        self._records: dict[str, _PoolRecord] = {}

    def get_state(self, pool: str) -> PoolStatus:
        """Return the current PoolStatus for a pool, defaulting to healthy."""
        rec = self._records.get(pool)
        if rec is None:
            return PoolStatus(pool=pool, state=HealthState.HEALTHY)
        return PoolStatus(
            pool=pool,
            state=rec.state,
            reason=rec.reason,
            override_until=rec.override_until,
        )

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus:
        """Mark a pool degraded for at least ttl_seconds without mutating deployment args."""
        now = self._now()
        rec = self._records.setdefault(pool, _PoolRecord())
        if rec.state != HealthState.UNHEALTHY:
            rec.state = HealthState.DEGRADED
            rec.reason = reason
        rec.override_until = max(rec.override_until, now + ttl_seconds)
        rec.last_transition_at = now
        rec.consecutive_ok = 0
        return self.get_state(pool)

    def record_probe(self, pool: str, *, ok: bool) -> PoolStatus:
        """Record an engine health probe and apply hysteresis before recovering."""
        now = self._now()
        rec = self._records.setdefault(pool, _PoolRecord())
        if not ok:
            rec.consecutive_ok = 0
            rec.consecutive_fail += 1
            if rec.consecutive_fail >= self._unhealthy_threshold:
                rec.state = HealthState.UNHEALTHY
                rec.reason = "probe_failures"
                rec.last_transition_at = now
            elif rec.state == HealthState.HEALTHY:
                rec.state = HealthState.DEGRADED
                rec.reason = "probe_degraded"
                rec.last_transition_at = now
            return self.get_state(pool)

        rec.consecutive_fail = 0
        if rec.state == HealthState.HEALTHY:
            return self.get_state(pool)

        if now < rec.override_until or (now - rec.last_transition_at) < self._recovery_cooldown:
            rec.consecutive_ok = 0
            return self.get_state(pool)

        rec.consecutive_ok += 1
        if rec.consecutive_ok >= self._min_healthy_probes:
            rec.state = HealthState.HEALTHY
            rec.reason = None
            rec.override_until = 0.0
            rec.last_transition_at = now
            rec.consecutive_ok = 0
        return self.get_state(pool)
