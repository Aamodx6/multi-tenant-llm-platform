from __future__ import annotations

import json
import os
from pathlib import Path

import redis
from starlette.applications import Starlette

from catalog.pool_catalog import ModelPoolCatalog, load_and_validate
from gateway.pool_health import PoolHealth, PoolHealthStore, RedisPoolHealthAdapter
from gateway.request_control import ReplicaEndpoint, VLLMAdapter, create_app
from gateway.tenant_budget import (
    BudgetAdapter,
    InMemoryBudgetAdapter,
    PriorityClass,
    RedisPostgresAdapter,
    TenantBudget,
    TenantRecord,
    hash_api_key,
)
from observability.metrics import MetricsRegistry


def _load_tenants() -> dict[str, TenantRecord]:
    raw_json = os.environ.get("TENANTS_JSON")
    if not raw_json:
        demo_paid = TenantRecord(
            "demo-paid", PriorityClass.PAID, tpm_limit=120_000, max_concurrency=16
        )
        demo_free = TenantRecord(
            "demo-free", PriorityClass.FREE, tpm_limit=20_000, max_concurrency=4
        )
        return {
            hash_api_key(os.environ.get("PAID_API_KEY", "sk-paid-demo")): demo_paid,
            hash_api_key(os.environ.get("FREE_API_KEY", "sk-free-demo")): demo_free,
        }
    parsed = json.loads(raw_json)
    return {
        hash_api_key(str(item["api_key"])): TenantRecord(
            tenant_id=str(item["tenant_id"]),
            priority=PriorityClass(str(item["priority"])),
            tpm_limit=int(item["tpm_limit"]),
            max_concurrency=int(item.get("max_concurrency", 8)),
        )
        for item in parsed
    }


def _load_replicas(catalog: ModelPoolCatalog) -> dict[str, list[ReplicaEndpoint]]:
    raw = os.environ.get("VLLM_ENDPOINTS")
    if not raw:
        return {
            name: [ReplicaEndpoint(replica_id=f"{name}-0", url=f"http://vllm-{name}:8000")]
            for name in catalog.pools
        }
    parsed = json.loads(raw)
    return {
        pool: [
            ReplicaEndpoint(replica_id=f"{pool}-{idx}", url=str(url))
            for idx, url in enumerate(urls)
        ]
        for pool, urls in parsed.items()
    }


def create_default_app() -> Starlette:
    """Build RequestControl app from environment variables and catalog/pools.yaml."""
    catalog_path = Path(os.environ.get("CATALOG_PATH", "catalog/pools.yaml"))
    catalog = load_and_validate(catalog_path.read_text(encoding="utf-8"))
    keys_by_hash = _load_tenants()

    redis_url = os.environ.get("REDIS_URL")
    budget_adapter: BudgetAdapter
    pool_health: PoolHealth
    if redis_url:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        budget_adapter = RedisPostgresAdapter(
            redis_client=client,
            pg_lookup=keys_by_hash.get,
            gateway_replicas=int(os.environ.get("GATEWAY_REPLICAS", "1")),
        )
        pool_health = RedisPoolHealthAdapter(redis_client=client)
    else:
        budget_adapter = InMemoryBudgetAdapter(keys_by_hash)
        pool_health = PoolHealthStore()

    poll_interval = float(os.environ.get("METRICS_POLL_INTERVAL_SECONDS", "5.0"))
    return create_app(
        catalog=catalog,
        budget=TenantBudget(adapter=budget_adapter),
        pool_health=pool_health,
        engine=VLLMAdapter(replicas=_load_replicas(catalog)),
        metrics=MetricsRegistry(),
        metrics_poll_interval_seconds=poll_interval if poll_interval > 0 else None,
    )


app = create_default_app()
