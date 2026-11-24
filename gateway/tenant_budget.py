from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

import redis


class AuthenticationError(PermissionError):
    """Raised when a presented API key is missing or unknown."""


class BudgetExceededError(RuntimeError):
    """Raised when a tenant exceeds its token-per-minute or concurrency quota."""

    def __init__(self, message: str, *, retry_after_seconds: int = 10) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class PriorityClass(StrEnum):
    """Tenant service tier governing admission order and shedding."""

    PAID = "paid"
    FREE = "free"


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


def hash_api_key(raw_key: str) -> str:
    """Return a SHA-256 hex digest for storing and comparing API keys."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _retry_after_seconds(requested: int, available: float, tpm_limit: int) -> int:
    deficit = requested - available
    rate_per_sec = max(tpm_limit / 60.0, 1e-6)
    return max(1, int(deficit / rate_per_sec) + 1)


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
    """In-process token-bucket adapter for deterministic unit and gateway tests."""

    def __init__(
        self,
        keys_by_hash: dict[str, TenantRecord],
        *,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keys = dict(keys_by_hash)
        self._now = now_fn
        self._buckets: dict[str, _Bucket] = {}

    def lookup_tenant(self, key_hash: str) -> TenantRecord | None:
        return self._keys.get(key_hash)

    def reserve(self, tenant: TenantRecord, model: str, est_tokens: int) -> Permit:
        now = self._now()
        bucket = self._refill(tenant, now)
        if bucket.tokens < est_tokens:
            retry = _retry_after_seconds(est_tokens, bucket.tokens, tenant.tpm_limit)
            raise BudgetExceededError(
                f"Tenant '{tenant.tenant_id}' exceeded TPM budget.",
                retry_after_seconds=retry,
            )
        bucket.tokens -= est_tokens
        return Permit(
            permit_id=uuid.uuid4().hex,
            tenant_id=tenant.tenant_id,
            priority=tenant.priority,
            model=model,
            reserved_tokens=est_tokens,
        )

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        delta = permit.reserved_tokens - actual_tokens
        bucket = self._buckets.get(permit.tenant_id)
        if bucket is None or delta == 0:
            return
        bucket.tokens = max(0.0, bucket.tokens + delta)

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
local key = KEYS[1]
local cap = tonumber(ARGV[1])
local rate = cap / 60.0
local now = tonumber(ARGV[2])
local req = tonumber(ARGV[3])

local vals = redis.call('HMGET', key, 'tokens', 'ts')
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
  redis.call('HMSET', key, 'tokens', tokens, 'ts', ts)
  return {0, tostring(tokens)}
end

tokens = tokens - req
redis.call('HMSET', key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', key, 120)
return {1, tostring(tokens)}
"""

_RECONCILE_LUA = """
local key = KEYS[1]
local delta = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local tokens = tonumber(redis.call('HGET', key, 'tokens'))
if tokens == nil then
  return 0
end
tokens = math.min(cap, math.max(0.0, tokens + delta))
redis.call('HSET', key, 'tokens', tokens)
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
        key_cache_ttl_seconds: int = 300,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._redis = redis_client
        self._pg_lookup = pg_lookup
        self._degraded_ratio = degraded_tpm_ratio
        self._key_cache_ttl = key_cache_ttl_seconds
        self._now = now_fn
        self._local_fallback = InMemoryBudgetAdapter({}, now_fn=now_fn)
        self._tenant_caps: dict[str, int] = {}

    def lookup_tenant(self, key_hash: str) -> TenantRecord | None:
        cache_key = f"tenant_key:{key_hash}"
        try:
            cached = self._redis.get(cache_key)
            if isinstance(cached, str):
                data: dict[str, Any] = json.loads(cached)
                return TenantRecord(
                    tenant_id=str(data["tenant_id"]),
                    priority=PriorityClass(str(data["priority"])),
                    tpm_limit=int(data["tpm_limit"]),
                    max_concurrency=int(data["max_concurrency"]),
                )
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

    def reserve(self, tenant: TenantRecord, model: str, est_tokens: int) -> Permit:
        self._tenant_caps[tenant.tenant_id] = tenant.tpm_limit
        bucket_key = f"tpm:{tenant.tenant_id}"
        now = self._now()
        try:
            res: Any = self._redis.eval(
                _RESERVE_LUA,
                1,
                bucket_key,
                str(tenant.tpm_limit),
                f"{now:.6f}",
                str(est_tokens),
            )
            allowed = int(res[0]) == 1
            remaining = float(res[1])
            if not allowed:
                retry = _retry_after_seconds(est_tokens, remaining, tenant.tpm_limit)
                raise BudgetExceededError(
                    f"Tenant '{tenant.tenant_id}' exceeded TPM budget.",
                    retry_after_seconds=retry,
                )
            return Permit(
                permit_id=uuid.uuid4().hex,
                tenant_id=tenant.tenant_id,
                priority=tenant.priority,
                model=model,
                reserved_tokens=est_tokens,
                degraded_mode=False,
            )
        except redis.RedisError:
            conservative = replace(
                tenant,
                tpm_limit=max(1, int(tenant.tpm_limit * self._degraded_ratio)),
            )
            permit = self._local_fallback.reserve(conservative, model, est_tokens)
            return replace(permit, degraded_mode=True)

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        if permit.degraded_mode:
            self._local_fallback.reconcile(permit, actual_tokens)
            return
        delta = permit.reserved_tokens - actual_tokens
        if delta == 0:
            return
        cap = self._tenant_caps.get(permit.tenant_id, max(permit.reserved_tokens, 1))
        try:
            self._redis.eval(
                _RECONCILE_LUA,
                1,
                f"tpm:{permit.tenant_id}",
                str(delta),
                str(cap),
            )
        except redis.RedisError:
            self._local_fallback.reconcile(permit, actual_tokens)


class TenantBudget:
    """Deep module encapsulating API key auth, token reservation, and usage reconciliation."""

    def __init__(self, adapter: BudgetAdapter) -> None:
        self._adapter = adapter

    def authorize_and_reserve(
        self, api_key: str, model: str, est_tokens: int
    ) -> Permit:
        """Authenticate an API key and reserve est_tokens against the tenant's bucket."""
        if not api_key:
            raise AuthenticationError("Missing API key.")
        tenant = self._adapter.lookup_tenant(hash_api_key(api_key))
        if tenant is None:
            raise AuthenticationError("Invalid API key.")
        return self._adapter.reserve(tenant, model, est_tokens)

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        """Reconcile actual streamed/completed tokens against the issued permit."""
        self._adapter.reconcile(permit, max(0, actual_tokens))
