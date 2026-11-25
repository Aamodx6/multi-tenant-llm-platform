from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

import redis
from sqlalchemy import create_engine, text

from gateway.db import normalize_database_url


class AuthenticationError(PermissionError):
    """Raised when a presented API key is missing or unknown."""


class PriorityClass(StrEnum):
    """Tenant service tier governing admission order and shedding."""

    PAID = "paid"
    FREE = "free"


class BudgetExceededError(RuntimeError):
    """Raised when a tenant exceeds its token-per-minute or concurrency quota."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int = 10,
        tenant_id: str = "unknown",
        priority: PriorityClass | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.tenant_id = tenant_id
        self.priority = priority


@dataclass(frozen=True)
class TenantRecord:
    """Authoritative tenant configuration and limits."""

    tenant_id: str
    priority: PriorityClass
    tpm_limit: int
    max_concurrency: int = 8


@dataclass(frozen=True)
class Permit:
    """Token reservation issued for a single request and reconciled after completion."""

    permit_id: str
    tenant_id: str
    priority: PriorityClass
    model: str
    reserved_tokens: int
    degraded_mode: bool = False
    tpm_limit: int = 0


def hash_api_key(raw_key: str) -> str:
    """Return a SHA-256 hex digest for storing and comparing API keys."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _retry_after_seconds(requested: int, available: float, tpm_limit: int) -> int:
    deficit = requested - available
    rate_per_sec = max(tpm_limit / 60.0, 1e-6)
    return max(1, int(deficit / rate_per_sec) + 1)


def _concurrency_exceeded_error(tenant: TenantRecord) -> BudgetExceededError:
    return BudgetExceededError(
        f"Tenant '{tenant.tenant_id}' exceeded concurrency cap ({tenant.max_concurrency}).",
        retry_after_seconds=1,
        tenant_id=tenant.tenant_id,
        priority=tenant.priority,
    )


class BudgetAdapter(Protocol):
    """Storage seam encapsulating key lookup, token-bucket state, and usage reconciliation."""

    def lookup_tenant(self, key_hash: str) -> TenantRecord | None: ...

    def reserve(
        self, tenant: TenantRecord, model: str, est_tokens: int
    ) -> Permit: ...

    def reconcile(self, permit: Permit, actual_tokens: int) -> None: ...


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class InMemoryBudgetAdapter:
    """In-process token-bucket and concurrency adapter for deterministic RequestControl tests."""

    def __init__(
        self,
        keys_by_hash: dict[str, TenantRecord],
        *,
        pg_lookup: Callable[[str], TenantRecord | None] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keys = dict(keys_by_hash)
        self._pg_lookup = pg_lookup
        self._now = now_fn
        self._buckets: dict[str, _Bucket] = {}
        self._caps: dict[str, int] = {}
        self._active_permits: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def lookup_tenant(self, key_hash: str) -> TenantRecord | None:
        if self._pg_lookup is not None:
            found = self._pg_lookup(key_hash)
            if found is not None:
                return found
        return self._keys.get(key_hash)

    def reserve(self, tenant: TenantRecord, model: str, est_tokens: int) -> Permit:
        with self._lock:
            active = self._active_permits.setdefault(tenant.tenant_id, set())
            if len(active) >= tenant.max_concurrency:
                raise _concurrency_exceeded_error(tenant)
            now = self._now()
            self._caps[tenant.tenant_id] = tenant.tpm_limit
            bucket = self._refill(tenant, now)
            if bucket.tokens < est_tokens:
                retry = _retry_after_seconds(est_tokens, bucket.tokens, tenant.tpm_limit)
                raise BudgetExceededError(
                    f"Tenant '{tenant.tenant_id}' exceeded TPM budget.",
                    retry_after_seconds=retry,
                    tenant_id=tenant.tenant_id,
                    priority=tenant.priority,
                )
            bucket.tokens -= est_tokens
            permit_id = uuid.uuid4().hex
            active.add(permit_id)
            return Permit(
                permit_id=permit_id,
                tenant_id=tenant.tenant_id,
                priority=tenant.priority,
                model=model,
                reserved_tokens=est_tokens,
                tpm_limit=tenant.tpm_limit,
            )

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        with self._lock:
            active = self._active_permits.get(permit.tenant_id)
            if active is not None:
                active.discard(permit.permit_id)
            delta = permit.reserved_tokens - actual_tokens
            bucket = self._buckets.get(permit.tenant_id)
            if bucket is None or delta == 0:
                return
            cap_int = permit.tpm_limit or self._caps.get(
                permit.tenant_id, max(permit.reserved_tokens, 1)
            )
            cap = float(cap_int)
            now = self._now()
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(cap, bucket.tokens + elapsed * (cap / 60.0))
            bucket.last_refill = now
            bucket.tokens = max(-4.0 * cap, min(cap, bucket.tokens + delta))

    def _refill(self, tenant: TenantRecord, now: float) -> _Bucket:
        bucket = self._buckets.get(tenant.tenant_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(tenant.tpm_limit), last_refill=now)
            self._buckets[tenant.tenant_id] = bucket
            return bucket
        elapsed = max(0.0, now - bucket.last_refill)
        refill = elapsed * (tenant.tpm_limit / 60.0)
        bucket.tokens = min(float(tenant.tpm_limit), bucket.tokens + refill)
        bucket.last_refill = now
        return bucket


_RESERVE_LUA = """
local bucket_key = KEYS[1]
local conc_key = KEYS[2]
local cap = tonumber(ARGV[1])
local now_override = ARGV[2]
local req = tonumber(ARGV[3])
local max_conc = tonumber(ARGV[4])
local permit_id = ARGV[5]
local permit_ttl = tonumber(ARGV[6])

local now
if now_override ~= '' then
  now = tonumber(now_override)
else
  local t = redis.call('TIME')
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000.0
end

redis.call('ZREMRANGEBYSCORE', conc_key, '-inf', now)
if redis.call('ZCARD', conc_key) >= max_conc then
  return {-1, tostring(max_conc)}
end

local rate = cap / 60.0
local vals = redis.call('HMGET', bucket_key, 'tokens', 'ts')
local tokens = tonumber(vals[1])
local ts = tonumber(vals[2])
if tokens == nil then
  tokens = cap
  ts = now
else
  local elapsed = math.max(0.0, now - ts)
  tokens = math.min(cap, tokens + elapsed * rate)
  ts = now
end

if tokens < req then
  redis.call('HSET', bucket_key, 'tokens', tokens, 'ts', ts)
  redis.call('EXPIRE', bucket_key, 120)
  return {0, tostring(tokens)}
end

tokens = tokens - req
redis.call('HSET', bucket_key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', bucket_key, 120)
redis.call('ZADD', conc_key, now + permit_ttl, permit_id)
redis.call('EXPIRE', conc_key, math.ceil(permit_ttl) + 30)
return {1, tostring(tokens)}
"""

_RECONCILE_LUA = """
local bucket_key = KEYS[1]
local conc_key = KEYS[2]
local delta = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local permit_id = ARGV[3]
local now_override = ARGV[4]

redis.call('ZREM', conc_key, permit_id)
if delta == 0 then
  return 1
end

local now
if now_override ~= '' then
  now = tonumber(now_override)
else
  local t = redis.call('TIME')
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000.0
end

local rate = math.max(cap / 60.0, 1e-6)
local vals = redis.call('HMGET', bucket_key, 'tokens', 'ts')
local tokens = tonumber(vals[1])
local ts = tonumber(vals[2])
if tokens == nil then
  if delta >= 0 then
    return 0
  end
  tokens = delta
  ts = now
else
  if ts ~= nil then
    local elapsed = math.max(0.0, now - ts)
    tokens = math.min(cap, tokens + elapsed * rate)
  end
  ts = now
  tokens = tokens + delta
end

tokens = math.max(-4.0 * cap, math.min(cap, tokens))
local ttl = 120
if tokens < 0 then
  ttl = math.max(120, math.ceil(-tokens / rate) + 60)
end
redis.call('HSET', bucket_key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', bucket_key, ttl)
return 1
"""


class RedisPostgresAdapter:
    """Production BudgetAdapter using Redis Lua buckets, Postgres lookup, and degraded fallback."""

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        pg_lookup: Callable[[str], TenantRecord | None],
        degraded_tpm_ratio: float = 0.25,
        gateway_replicas: int = 1,
        key_cache_ttl_seconds: int = 300,
        permit_ttl_seconds: float = 600.0,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._redis = redis_client
        self._pg_lookup = pg_lookup
        self._degraded_ratio = degraded_tpm_ratio
        self._gateway_replicas = max(1, gateway_replicas)
        self._key_cache_ttl = key_cache_ttl_seconds
        self._permit_ttl = permit_ttl_seconds
        self._now = now_fn
        self._local_fallback = InMemoryBudgetAdapter(
            {},
            now_fn=now_fn or time.monotonic,
        )
        self._tenant_caps: dict[str, int] = {}
        self._pending_reconciles: deque[tuple[Permit, int]] = deque(maxlen=1024)
        self._pending_lock = threading.Lock()

    def lookup_tenant(self, key_hash: str) -> TenantRecord | None:
        cache_key = f"tenant_key:{key_hash}"
        try:
            cached = self._redis.get(cache_key)
            if isinstance(cached, str):
                data: dict[str, Any] = json.loads(cached)
                rec = TenantRecord(
                    tenant_id=str(data["tenant_id"]),
                    priority=PriorityClass(str(data["priority"])),
                    tpm_limit=int(data["tpm_limit"]),
                    max_concurrency=int(data["max_concurrency"]),
                )
                self._tenant_caps[rec.tenant_id] = rec.tpm_limit
                return rec
        except redis.RedisError:
            pass

        tenant = self._pg_lookup(key_hash)
        if tenant is None:
            return None
        self._tenant_caps[tenant.tenant_id] = tenant.tpm_limit
        try:
            payload = json.dumps(
                {
                    "tenant_id": tenant.tenant_id,
                    "priority": str(tenant.priority),
                    "tpm_limit": tenant.tpm_limit,
                    "max_concurrency": tenant.max_concurrency,
                }
            )
            self._redis.set(cache_key, payload, ex=self._key_cache_ttl)
        except redis.RedisError:
            pass
        return tenant

    def _eval_reconcile(self, permit: Permit, actual_tokens: int) -> None:
        delta = permit.reserved_tokens - actual_tokens
        cap = (
            permit.tpm_limit
            or self._tenant_caps.get(permit.tenant_id)
            or max(permit.reserved_tokens, 1)
        )
        now_override = f"{self._now():.6f}" if self._now is not None else ""
        self._redis.eval(
            _RECONCILE_LUA,
            2,
            f"tpm:{permit.tenant_id}",
            f"conc:{permit.tenant_id}",
            str(delta),
            str(cap),
            permit.permit_id,
            now_override,
        )

    def _flush_pending_reconciles(self) -> None:
        with self._pending_lock:
            while self._pending_reconciles:
                pending_permit, pending_actual = self._pending_reconciles[0]
                self._eval_reconcile(pending_permit, pending_actual)
                self._pending_reconciles.popleft()

    def reserve(self, tenant: TenantRecord, model: str, est_tokens: int) -> Permit:
        self._tenant_caps[tenant.tenant_id] = tenant.tpm_limit
        permit_id = uuid.uuid4().hex
        now_override = f"{self._now():.6f}" if self._now is not None else ""
        try:
            self._flush_pending_reconciles()
            res: Any = self._redis.eval(
                _RESERVE_LUA,
                2,
                f"tpm:{tenant.tenant_id}",
                f"conc:{tenant.tenant_id}",
                str(tenant.tpm_limit),
                now_override,
                str(est_tokens),
                str(tenant.max_concurrency),
                permit_id,
                f"{self._permit_ttl:.1f}",
            )
            code = int(res[0])
            if code == -1:
                raise _concurrency_exceeded_error(tenant)
            if code == 0:
                remaining = float(res[1])
                retry = _retry_after_seconds(est_tokens, remaining, tenant.tpm_limit)
                raise BudgetExceededError(
                    f"Tenant '{tenant.tenant_id}' exceeded TPM budget.",
                    retry_after_seconds=retry,
                    tenant_id=tenant.tenant_id,
                    priority=tenant.priority,
                )
            return Permit(
                permit_id=permit_id,
                tenant_id=tenant.tenant_id,
                priority=tenant.priority,
                model=model,
                reserved_tokens=est_tokens,
                degraded_mode=False,
                tpm_limit=tenant.tpm_limit,
            )
        except redis.RedisError:
            conservative = replace(
                tenant,
                tpm_limit=max(
                    1,
                    int(tenant.tpm_limit * self._degraded_ratio / self._gateway_replicas),
                ),
                max_concurrency=max(1, tenant.max_concurrency // self._gateway_replicas),
            )
            permit = self._local_fallback.reserve(conservative, model, est_tokens)
            return replace(permit, degraded_mode=True)

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        if permit.degraded_mode:
            self._local_fallback.reconcile(permit, actual_tokens)
            return
        try:
            self._flush_pending_reconciles()
            self._eval_reconcile(permit, actual_tokens)
        except redis.RedisError:
            with self._pending_lock:
                self._pending_reconciles.append((permit, actual_tokens))


def build_pg_lookup(
    database_url: str, fallback: dict[str, TenantRecord] | None = None
) -> Callable[[str], TenantRecord | None]:
    """Create a live SQLAlchemy tenant lookup function backed by the tenants table."""
    engine = create_engine(normalize_database_url(database_url), pool_pre_ping=True)
    static_map = dict(fallback or {})

    def _lookup(key_hash: str) -> TenantRecord | None:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT tenant_id, priority, tpm_limit, max_concurrency "
                    "FROM tenants WHERE key_hash = :key_hash AND revoked_at IS NULL"
                ),
                {"key_hash": key_hash},
            ).fetchone()
        if row is not None:
            return TenantRecord(
                tenant_id=str(row[0]),
                priority=PriorityClass(str(row[1])),
                tpm_limit=int(row[2]),
                max_concurrency=int(row[3]),
            )
        return static_map.get(key_hash)

    return _lookup


class TenantBudget:
    """Deep module encapsulating API key auth, concurrency & token limits, and reconciliation."""

    def __init__(self, adapter: BudgetAdapter) -> None:
        self._adapter = adapter

    def authenticate(self, api_key: str) -> TenantRecord:
        """Authenticate a raw Bearer API key and return its TenantRecord."""
        if not api_key:
            raise AuthenticationError("Missing API key.")
        tenant = self._adapter.lookup_tenant(hash_api_key(api_key))
        if tenant is None:
            raise AuthenticationError("Invalid API key.")
        return tenant

    def reserve(self, tenant: TenantRecord, model: str, est_tokens: int) -> Permit:
        """Reserve concurrency + est_tokens against an already-authenticated TenantRecord."""
        return self._adapter.reserve(tenant, model, est_tokens)

    def authorize_and_reserve(
        self, api_key: str, model: str, est_tokens: int
    ) -> Permit:
        """Authenticate an API key and reserve concurrency + est_tokens against the tenant."""
        tenant = self.authenticate(api_key)
        return self.reserve(tenant, model, est_tokens)

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        """Release concurrency slot and reconcile actual tokens against the issued permit."""
        self._adapter.reconcile(permit, max(0, actual_tokens))
