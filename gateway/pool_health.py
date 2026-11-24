from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import redis


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
    probe_degraded: bool = False


def _status_for(pool: str, rec: _PoolRecord, now: float) -> PoolStatus:
    if (
        rec.state == HealthState.DEGRADED
        and not rec.probe_degraded
        and rec.override_until > 0.0
        and now >= rec.override_until
    ):
        rec.state = HealthState.HEALTHY
        rec.reason = None
        rec.override_until = 0.0
        rec.consecutive_ok = 0
    return PoolStatus(
        pool=pool,
        state=rec.state,
        reason=rec.reason,
        override_until=rec.override_until,
    )


def _apply_degraded(rec: _PoolRecord, reason: str, ttl_seconds: float, now: float) -> None:
    if rec.state != HealthState.UNHEALTHY:
        rec.state = HealthState.DEGRADED
        rec.reason = reason
    rec.override_until = max(rec.override_until, now + ttl_seconds)
    rec.last_transition_at = now
    rec.consecutive_ok = 0


def _apply_probe(
    rec: _PoolRecord,
    *,
    ok: bool,
    now: float,
    recovery_cooldown: float,
    min_healthy_probes: int,
    unhealthy_threshold: int,
) -> None:
    if not ok:
        rec.consecutive_ok = 0
        rec.consecutive_fail += 1
        rec.probe_degraded = True
        if rec.consecutive_fail >= unhealthy_threshold:
            rec.state = HealthState.UNHEALTHY
            rec.reason = "probe_failures"
            rec.last_transition_at = now
        elif rec.state == HealthState.HEALTHY:
            rec.state = HealthState.DEGRADED
            rec.reason = "probe_degraded"
            rec.last_transition_at = now
        return

    rec.consecutive_fail = 0
    if rec.state == HealthState.HEALTHY:
        return

    if now < rec.override_until or (now - rec.last_transition_at) < recovery_cooldown:
        rec.consecutive_ok = 0
        return

    rec.consecutive_ok += 1
    if rec.consecutive_ok >= min_healthy_probes:
        rec.state = HealthState.HEALTHY
        rec.reason = None
        rec.override_until = 0.0
        rec.last_transition_at = now
        rec.consecutive_ok = 0
        rec.probe_degraded = False


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
        """Return current PoolStatus, auto-expiring TTL-only degraded overrides."""
        rec = self._records.get(pool)
        if rec is None:
            return PoolStatus(pool=pool, state=HealthState.HEALTHY)
        return _status_for(pool, rec, self._now())

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus:
        """Mark a pool degraded for ttl_seconds without mutating deployment args."""
        now = self._now()
        rec = self._records.setdefault(pool, _PoolRecord())
        _apply_degraded(rec, reason, ttl_seconds, now)
        return _status_for(pool, rec, now)

    def record_probe(self, pool: str, *, ok: bool) -> PoolStatus:
        """Record an engine health probe and apply hysteresis before recovering."""
        now = self._now()
        rec = self._records.setdefault(pool, _PoolRecord())
        _apply_probe(
            rec,
            ok=ok,
            now=now,
            recovery_cooldown=self._recovery_cooldown,
            min_healthy_probes=self._min_healthy_probes,
            unhealthy_threshold=self._unhealthy_threshold,
        )
        return _status_for(pool, rec, now)


class RedisPoolHealthAdapter:
    """Redis-backed PoolHealth adapter shared across RequestControl and controller pods."""

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        recovery_cooldown_seconds: float = 30.0,
        min_healthy_probes: int = 2,
        unhealthy_failure_threshold: int = 3,
        state_ttl_seconds: int = 3600,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._redis = redis_client
        self._recovery_cooldown = recovery_cooldown_seconds
        self._min_healthy_probes = min_healthy_probes
        self._unhealthy_threshold = unhealthy_failure_threshold
        self._state_ttl = state_ttl_seconds
        self._now = now_fn
        self._local_fallback = PoolHealthStore(
            recovery_cooldown_seconds=recovery_cooldown_seconds,
            min_healthy_probes=min_healthy_probes,
            unhealthy_failure_threshold=unhealthy_failure_threshold,
            now_fn=now_fn,
        )

    def get_state(self, pool: str) -> PoolStatus:
        """Read shared PoolStatus from Redis, auto-expiring TTL overrides."""
        now = self._now()
        try:
            rec = self._load_record(pool)
        except redis.RedisError:
            return self._local_fallback.get_state(pool)
        if rec is None:
            return PoolStatus(pool=pool, state=HealthState.HEALTHY)
        status = _status_for(pool, rec, now)
        if status.state == HealthState.HEALTHY and rec.last_transition_at > 0.0:
            with contextlib.suppress(redis.RedisError):
                self._save_record(pool, rec)
        return status

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus:
        """Persist a TTL-bounded degraded override in Redis across all pods."""
        now = self._now()
        try:
            rec = self._load_record(pool) or _PoolRecord()
            _apply_degraded(rec, reason, ttl_seconds, now)
            self._save_record(pool, rec)
            return _status_for(pool, rec, now)
        except redis.RedisError:
            return self._local_fallback.set_degraded(pool, reason, ttl_seconds)

    def record_probe(self, pool: str, *, ok: bool) -> PoolStatus:
        """Update probe-based hysteresis state in Redis."""
        now = self._now()
        try:
            rec = self._load_record(pool) or _PoolRecord()
            _apply_probe(
                rec,
                ok=ok,
                now=now,
                recovery_cooldown=self._recovery_cooldown,
                min_healthy_probes=self._min_healthy_probes,
                unhealthy_threshold=self._unhealthy_threshold,
            )
            self._save_record(pool, rec)
            return _status_for(pool, rec, now)
        except redis.RedisError:
            return self._local_fallback.record_probe(pool, ok=ok)

    def _load_record(self, pool: str) -> _PoolRecord | None:
        raw = self._redis.get(f"pool_health:{pool}")
        if not isinstance(raw, str):
            return None
        data: dict[str, Any] = json.loads(raw)
        return _PoolRecord(
            state=HealthState(str(data["state"])),
            reason=data.get("reason"),
            override_until=float(data.get("override_until", 0.0)),
            last_transition_at=float(data.get("last_transition_at", 0.0)),
            consecutive_ok=int(data.get("consecutive_ok", 0)),
            consecutive_fail=int(data.get("consecutive_fail", 0)),
            probe_degraded=bool(data.get("probe_degraded", False)),
        )

    def _save_record(self, pool: str, rec: _PoolRecord) -> None:
        payload = json.dumps(
            {
                "state": str(rec.state),
                "reason": rec.reason,
                "override_until": rec.override_until,
                "last_transition_at": rec.last_transition_at,
                "consecutive_ok": rec.consecutive_ok,
                "consecutive_fail": rec.consecutive_fail,
                "probe_degraded": rec.probe_degraded,
            }
        )
        self._redis.set(f"pool_health:{pool}", payload, ex=self._state_ttl)
