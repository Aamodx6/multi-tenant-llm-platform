import json

import httpx
import pytest
import yaml

from catalog.pool_catalog import load_and_validate
from controller.self_healing import (
    GuardrailState,
    PodTelemetry,
    SelfHealingEngine,
    SimulatedActuatorAdapter,
    TelemetryWindow,
)
from gateway.pool_health import PoolHealthStore
from gateway.request_control import FakeEngineAdapter, ReplicaEndpoint, VLLMAdapter, create_app
from gateway.tenant_budget import (
    InMemoryBudgetAdapter,
    PriorityClass,
    TenantBudget,
    TenantRecord,
    hash_api_key,
)
from observability.bundle import render_observability_bundle
from observability.metrics import MetricsRegistry

CATALOG_YAML = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    fallback_pool: pool-fallback
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases: [qwen-7b]
  pool-fallback:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    isolation: fallback
    min_replicas: 0
    max_replicas: 2
    keda_waiting_threshold: 6
    keda_kv_cache_threshold: 0.85
    slo_p95_ttft_ms: 600
    slo_availability: 0.99
"""


@pytest.mark.anyio
async def test_metrics_endpoint_exposes_gateway_fallback_429_and_aiops_metrics() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    metrics = MetricsRegistry()
    health = PoolHealthStore()
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=50)
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(adapter=InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant})),
        pool_health=health,
        engine=FakeEngineAdapter(
            replicas={
                "pool-8b": [ReplicaEndpoint("8b-0", "http://8b-0:8000")],
                "pool-fallback": [ReplicaEndpoint("fb-0", "http://fb-0:8000")],
            },
            completion_tokens=10,
        ),
        metrics=metrics,
    )
    healer = SelfHealingEngine(
        actuator=SimulatedActuatorAdapter(),
        pool_health=health,
        metrics=metrics,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        ok = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert ok.status_code == 200

        # Trigger 429 rate limit
        over = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 45,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert over.status_code == 429

        # Trigger SelfHealingEngine degradation -> fallback on next call
        healer.reconcile(
            TelemetryWindow(
                timestamp=100.0,
                pods=[
                    PodTelemetry(
                        pod="8b-0",
                        node="n1",
                        pool="pool-8b",
                        kv_cache_usage=0.97,
                        gpu_temp_c=70.0,
                        ttft_ms_series=[100.0, 102.0, 99.0, 450.0],
                    )
                ],
            ),
            GuardrailState(),
        )
        fb = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert fb.status_code == 200
        assert fb.headers["x-served-pool"] == "pool-fallback"

        scrape = await client.get("/metrics")
        assert scrape.status_code == 200
        text = scrape.text
        assert "gateway_requests_total" in text
        assert "gateway_429_total" in text
        assert "gateway_fallback_total" in text
        assert "gateway_tokens_total" in text
        assert "aiops_anomaly_score" in text
        assert "aiops_remediation_actions_total" in text


@pytest.mark.anyio
async def test_lifespan_runs_background_vllm_metrics_scraper() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    rep = ReplicaEndpoint("8b-0", "http://vllm-0:8000")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"} 5\n'
            'vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct"} 0.65\n',
        )

    adapter = VLLMAdapter(
        replicas={"pool-8b": [rep]},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(adapter=InMemoryBudgetAdapter({})),
        pool_health=PoolHealthStore(),
        engine=adapter,
        metrics=MetricsRegistry(),
        metrics_poll_interval_seconds=0.05,
    )
    async with app.router.lifespan_context(app):
        import asyncio

        await asyncio.sleep(0.12)
    assert rep.requests_waiting == 5
    assert rep.kv_cache_usage == pytest.approx(0.65)


def test_render_observability_bundle_generates_valid_alerts_dcgm_and_grafana() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    bundle = render_observability_bundle(catalog)

    expected = {
        "prometheus-alerts.yaml",
        "dcgm-counters.csv",
        "grafana-dashboard.json",
        "prometheus-scrape.yaml",
    }
    assert expected <= set(bundle)
    alerts = yaml.safe_load(bundle["prometheus-alerts.yaml"])
    rules = [r for g in alerts["groups"] for r in g["rules"]]
    alert_names = {r["alert"] for r in rules}
    assert any("TTFT" in name for name in alert_names)
    assert any("BurnRate" in name for name in alert_names)
    assert "DCGMXidError" in alert_names

    dcgm_csv = bundle["dcgm-counters.csv"]
    assert "DCGM_FI_DEV_GPU_TEMP" in dcgm_csv
    assert "DCGM_FI_DEV_XID_ERRORS" in dcgm_csv
    assert "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL" in dcgm_csv

    dashboard = json.loads(bundle["grafana-dashboard.json"])
    assert len(dashboard["panels"]) >= 4
