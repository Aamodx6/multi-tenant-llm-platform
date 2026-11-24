from __future__ import annotations

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


def _status_for(pool: str, rec: _PoolRecord, now: float) -> tuple[PoolStatus, bool]:
    expired = False
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
        expired = True
    return (
        PoolStatus(
            pool=pool,
            state=rec.state,
            reason=rec.reason,
            override_until=rec.override_until,
        ),
        expired,
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
        status, _ = _status_for(pool, rec, self._now())
        return status

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus:
        """Mark a pool degraded for ttl_seconds without mutating deployment args."""
        now = self._now()
        rec = self._records.setdefault(pool, _PoolRecord())
        _apply_degraded(rec, reason, ttl_seconds, now)
        status, _ = _status_for(pool, rec, now)
        return status

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
        status, _ = _status_for(pool, rec, now)
        return status


_HEALTH_LUA_TIME = """
local now
if now_override ~= '' then
  now = tonumber(now_override)
else
  local t = redis.call('TIME')
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000.0
end
"""

_GET_STATE_LUA = (
    "local key = KEYS[1]\nlocal now_override = ARGV[1]\nlocal ttl = tonumber(ARGV[2])\n"
    + _HEALTH_LUA_TIME
    + """
local vals = redis.call('HMGET', key, 'state', 'reason', 'override_until', 'probe_degraded')
local state = vals[1]
if not state then
  return {'healthy', '', '0'}
end
local reason = vals[2] or ''
local override_until = tonumber(vals[3] or '0')
local probe_degraded = tonumber(vals[4] or '0')
if state == 'degraded' and probe_degraded == 0 and override_until > 0 and now >= override_until then
  redis.call(
    'HSET', key, 'state', 'healthy', 'reason', '', 'override_until', '0', 'consecutive_ok', '0'
  )
  redis.call('EXPIRE', key, ttl)
  return {'healthy', '', '0'}
end
return {state, reason, tostring(override_until)}
"""
)

_SET_DEGRADED_LUA = (
    "local key = KEYS[1]\nlocal reason = ARGV[1]\nlocal ttl_sec = tonumber(ARGV[2])\n"
    "local now_override = ARGV[3]\nlocal state_ttl = tonumber(ARGV[4])\n"
    + _HEALTH_LUA_TIME
    + """
local vals = redis.call('HMGET', key, 'state', 'reason', 'override_until')
local state = vals[1] or 'healthy'
local cur_reason = vals[2] or ''
local cur_until = tonumber(vals[3] or '0')
if state ~= 'unhealthy' then
  state = 'degraded'
  cur_reason = reason
end
local new_until = math.max(cur_until, now + ttl_sec)
redis.call('HSET', key,
  'state', state,
  'reason', cur_reason,
  'override_until', tostring(new_until),
  'last_transition_at', tostring(now),
  'consecutive_ok', '0'
)
redis.call('EXPIRE', key, state_ttl)
return {state, cur_reason, tostring(new_until)}
"""
)

_RECORD_PROBE_LUA = (
    "local key = KEYS[1]\nlocal ok = tonumber(ARGV[1])\nlocal cooldown = tonumber(ARGV[2])\n"
    "local min_ok = tonumber(ARGV[3])\nlocal fail_thresh = tonumber(ARGV[4])\n"
    "local now_override = ARGV[5]\nlocal state_ttl = tonumber(ARGV[6])\n"
    + _HEALTH_LUA_TIME
    + """
local vals = redis.call('HMGET', key,
  'state', 'reason', 'override_until', 'last_transition_at',
  'consecutive_ok', 'consecutive_fail', 'probe_degraded')
local state = vals[1] or 'healthy'
local reason = vals[2] or ''
local override_until = tonumber(vals[3] or '0')
local last_trans = tonumber(vals[4] or '0')
local cons_ok = tonumber(vals[5] or '0')
local cons_fail = tonumber(vals[6] or '0')
local probe_deg = tonumber(vals[7] or '0')

if ok == 0 then
  cons_ok = 0
  cons_fail = cons_fail + 1
  probe_deg = 1
  if cons_fail >= fail_thresh then
    state = 'unhealthy'
    reason = 'probe_failures'
    last_trans = now
  elseif state == 'healthy' then
    state = 'degraded'
    reason = 'probe_degraded'
    last_trans = now
  end
else
  cons_fail = 0
  if state ~= 'healthy' then
    if now < override_until or (now - last_trans) < cooldown then
      cons_ok = 0
    else
      cons_ok = cons_ok + 1
      if cons_ok >= min_ok then
        state = 'healthy'
        reason = ''
        override_until = 0
        last_trans = now
        cons_ok = 0
        probe_deg = 0
      end
    end
  end
end

redis.call('HSET', key,
  'state', state,
  'reason', reason,
  'override_until', tostring(override_until),
  'last_transition_at', tostring(last_trans),
  'consecutive_ok', tostring(cons_ok),
  'consecutive_fail', tostring(cons_fail),
  'probe_degraded', tostring(probe_deg)
)
redis.call('EXPIRE', key, state_ttl)
return {state, reason, tostring(override_until)}
"""
)


def _parse_lua_status(pool: str, raw: Any) -> PoolStatus:
    state = HealthState(str(raw[0]))
    reason = str(raw[1]) if raw[1] else None
    override_until = float(raw[2])
    return PoolStatus(pool=pool, state=state, reason=reason, override_until=override_until)


class RedisPoolHealthAdapter:
    """Atomic Lua Redis-backed PoolHealth adapter shared across all pods."""

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        recovery_cooldown_seconds: float = 30.0,
        min_healthy_probes: int = 2,
        unhealthy_failure_threshold: int = 3,
        state_ttl_seconds: int = 3600,
        now_fn: Callable[[], float] | None = None,
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
            now_fn=now_fn or time.monotonic,
        )

    def _now_arg(self) -> str:
        return f"{self._now():.6f}" if self._now is not None else ""

    def get_state(self, pool: str) -> PoolStatus:
        """Read shared PoolStatus atomically from Redis."""
        try:
            res = self._redis.eval(
                _GET_STATE_LUA,
                1,
                f"pool_health:{pool}",
                self._now_arg(),
                str(self._state_ttl),
            )
            return _parse_lua_status(pool, res)
        except redis.RedisError:
            return self._local_fallback.get_state(pool)

    def set_degraded(self, pool: str, reason: str, ttl_seconds: float) -> PoolStatus:
        """Atomically persist a TTL-bounded degraded override in Redis."""
        try:
            res = self._redis.eval(
                _SET_DEGRADED_LUA,
                1,
                f"pool_health:{pool}",
                reason,
                f"{ttl_seconds:.6f}",
                self._now_arg(),
                str(self._state_ttl),
            )
            return _parse_lua_status(pool, res)
        except redis.RedisError:
            return self._local_fallback.set_degraded(pool, reason, ttl_seconds)

    def record_probe(self, pool: str, *, ok: bool) -> PoolStatus:
        """Atomically evaluate probe-based hysteresis in Redis."""
        try:
            res = self._redis.eval(
                _RECORD_PROBE_LUA,
                1,
                f"pool_health:{pool}",
                "1" if ok else "0",
                f"{self._recovery_cooldown:.6f}",
                str(self._min_healthy_probes),
                str(self._unhealthy_threshold),
                self._now_arg(),
                str(self._state_ttl),
            )
            return _parse_lua_status(pool, res)
        except redis.RedisError:
            return self._local_fallback.record_probe(pool, ok=ok)
