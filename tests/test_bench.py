from pathlib import Path

import httpx
import pytest

from bench.cli import load_baseline_report
from bench.runner import (
    ArmComparison,
    BenchmarkReport,
    BenchmarkRunner,
    WorkloadProfile,
    compare_arms,
)
from catalog.pool_catalog import load_and_validate
from gateway.pool_health import PoolHealthStore
from gateway.request_control import FakeEngineAdapter, ReplicaEndpoint, create_app
from gateway.tenant_budget import (
    InMemoryBudgetAdapter,
    PriorityClass,
    TenantBudget,
    TenantRecord,
    hash_api_key,
)

CATALOG_YAML = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases: [qwen-7b]
"""


@pytest.mark.anyio
async def test_benchmark_runner_collects_sse_metrics_cost_and_checks_regression() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="bench", priority=PriorityClass.PAID, tpm_limit=100_000)
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(adapter=InMemoryBudgetAdapter({hash_api_key("sk-bench"): tenant})),
        pool_health=PoolHealthStore(),
        engine=FakeEngineAdapter(
            replicas={"pool-8b": [ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000")]},
            stream_chunks=["a", "b", "c", "d"],
        ),
    )

    profile = WorkloadProfile(
        name="prefix-cache-on",
        base_url="http://test",
        api_key="sk-bench",
        model="qwen-7b",
        num_requests=6,
        concurrency=2,
        max_tokens=32,
        stream=True,
        gpu_hourly_usd=0.72,
        slo_p95_ttft_ms=1000.0,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        runner = BenchmarkRunner(client=client)
        verdict = await runner.run(profile)

    r = verdict.report
    assert verdict.passed is True
    assert r.succeeded == 6
    assert r.failed == 0
    assert r.output_tokens_per_sec > 0.0
    assert r.total_tokens_per_sec > r.output_tokens_per_sec
    assert r.ttft_ms_p50 >= 0.0
    assert r.ttft_ms_p95 >= r.ttft_ms_p50
    assert r.itl_ms_p95 >= 0.0
    assert r.goodput_rps > 0.0
    expected_out_cost = (0.72 / (3600.0 * r.output_tokens_per_sec)) * 1_000_000.0
    assert r.cost_per_1m_output_usd == pytest.approx(expected_out_cost, rel=1e-4)

    # Regression gate: faster baseline triggers FAIL when current p95 TTFT regresses >15%
    fast_baseline = BenchmarkReport(
        profile_name="baseline",
        model="qwen-7b",
        total_requests=6,
        succeeded=6,
        failed=0,
        rate_limited_429=0,
        duration_seconds=0.1,
        requests_per_sec=60.0,
        output_tokens_per_sec=max(r.output_tokens_per_sec * 2.0, 100.0),
        total_tokens_per_sec=max(r.total_tokens_per_sec * 2.0, 200.0),
        ttft_ms_p50=0.001,
        ttft_ms_p95=0.001,
        ttft_ms_p99=0.001,
        itl_ms_p50=0.001,
        itl_ms_p95=0.001,
        e2el_ms_p50=0.01,
        e2el_ms_p95=0.01,
        goodput_rps=60.0,
        cost_per_1m_output_usd=0.5,
        cost_per_1m_total_usd=0.25,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        regressed = await BenchmarkRunner(client=client).run(profile, baseline=fast_baseline)
    assert regressed.passed is False
    assert regressed.reasons

    delta = compare_arms(fast_baseline, r)
    assert isinstance(delta, ArmComparison)
    assert delta.ttft_p95_delta_ms >= 0.0


def test_load_baseline_report_accepts_verdict_and_report_json(tmp_path: Path) -> None:
    report = BenchmarkReport(
        profile_name="p",
        model="qwen-7b",
        total_requests=1,
        succeeded=1,
        failed=0,
        rate_limited_429=0,
        duration_seconds=1.0,
        requests_per_sec=1.0,
        output_tokens_per_sec=10.0,
        total_tokens_per_sec=20.0,
        ttft_ms_p50=100.0,
        ttft_ms_p95=120.0,
        ttft_ms_p99=130.0,
        itl_ms_p50=10.0,
        itl_ms_p95=15.0,
        e2el_ms_p50=200.0,
        e2el_ms_p95=250.0,
        goodput_rps=1.0,
        cost_per_1m_output_usd=10.0,
        cost_per_1m_total_usd=5.0,
    )
    verdict_file = tmp_path / "verdict.json"
    verdict_file.write_text(
        f'{{"report": {report.model_dump_json()}, "passed": true, "reasons": []}}',
        encoding="utf-8",
    )
    loaded = load_baseline_report(verdict_file)
    assert loaded.ttft_ms_p95 == 120.0


@pytest.mark.anyio
async def test_benchmark_runner_fails_when_error_rate_exceeds_min_success_rate() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )
        return httpx.Response(503, json={"error": {"message": "down"}})

    profile = WorkloadProfile(
        name="error-gate",
        base_url="http://test",
        api_key="sk-bench",
        model="qwen-7b",
        num_requests=5,
        concurrency=1,
        stream=False,
        slo_p95_ttft_ms=1000.0,
        min_success_rate=0.95,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        verdict = await BenchmarkRunner(client=client).run(profile)

    assert verdict.report.succeeded == 1
    assert verdict.report.failed == 4
    assert verdict.passed is False
    assert any("success rate" in r.lower() for r in verdict.reasons)
