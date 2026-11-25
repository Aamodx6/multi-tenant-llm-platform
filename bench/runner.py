from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field


class WorkloadProfile(BaseModel):
    """Declarative benchmark workload and regression gate specification."""

    name: str
    base_url: str = "http://127.0.0.1:8080"
    api_key: str
    model: str = "qwen-7b"
    num_requests: int = Field(default=50, ge=1)
    concurrency: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=256, ge=1)
    warmup_requests: int = Field(default=0, ge=0)
    stream: bool = True
    system_prompt: str = "You are a concise engineering assistant."
    user_prompt_prefix: str = "Explain Kubernetes scheduling for GPU workloads"
    gpu_hourly_usd: float = Field(default=0.50, gt=0.0)
    slo_p95_ttft_ms: float = Field(default=1000.0, gt=0.0)
    min_success_rate: float = Field(default=0.99, ge=0.0, le=1.0)
    max_itl_p95_ms: float | None = Field(default=None, gt=0.0)
    max_ttft_regression_pct: float = Field(default=15.0, ge=0.0)
    min_throughput_ratio: float = Field(default=0.85, gt=0.0, le=1.0)


class ArmComparison(BaseModel):
    """Typed A/B comparison deltas between a baseline and candidate BenchmarkReport."""

    ttft_p95_delta_ms: float
    ttft_p95_speedup_pct: float
    output_tok_s_delta: float
    cost_1m_output_delta_usd: float


class BenchmarkReport(BaseModel):
    """Measured latency, throughput, goodput, and cost metrics from one run."""

    profile_name: str
    model: str
    total_requests: int
    succeeded: int
    failed: int
    rate_limited_429: int
    duration_seconds: float
    requests_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    ttft_ms_p99: float
    itl_ms_p50: float
    itl_ms_p95: float
    e2el_ms_p50: float
    e2el_ms_p95: float
    goodput_rps: float
    cost_per_1m_output_usd: float
    cost_per_1m_total_usd: float


class BenchmarkVerdict(BaseModel):
    """Pass/fail verdict against SLO bounds and an optional baseline report."""

    report: BenchmarkReport
    passed: bool
    reasons: list[str] = Field(default_factory=list)


def cost_per_million_tokens(gpu_hourly_usd: float, tok_per_sec: float) -> float:
    """Calculate $/1M tokens from hourly GPU cost and steady-state token throughput."""
    if tok_per_sec <= 0.0:
        return 0.0
    return (gpu_hourly_usd / (3600.0 * tok_per_sec)) * 1_000_000.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def compare_arms(baseline: BenchmarkReport, candidate: BenchmarkReport) -> ArmComparison:
    """Compute A/B latency, throughput, and cost deltas between two BenchmarkReports."""
    ttft_delta = candidate.ttft_ms_p95 - baseline.ttft_ms_p95
    speedup_pct = (
        ((baseline.ttft_ms_p95 - candidate.ttft_ms_p95) / baseline.ttft_ms_p95) * 100.0
        if baseline.ttft_ms_p95 > 0.0
        else 0.0
    )
    return ArmComparison(
        ttft_p95_delta_ms=ttft_delta,
        ttft_p95_speedup_pct=speedup_pct,
        output_tok_s_delta=candidate.output_tokens_per_sec - baseline.output_tokens_per_sec,
        cost_1m_output_delta_usd=(
            candidate.cost_per_1m_output_usd - baseline.cost_per_1m_output_usd
        ),
    )


@dataclass
class _RequestSample:
    ok: bool
    rate_limited: bool
    ttft_ms: float
    e2el_ms: float
    itl_ms: list[float]
    prompt_tokens: int
    completion_tokens: int


class BenchmarkRunner:
    """Concurrent OpenAI HTTP/SSE load driver with SLO & CI regression gating."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        now_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._now = now_fn

    async def run(
        self,
        profile: WorkloadProfile,
        baseline: BenchmarkReport | None = None,
    ) -> BenchmarkVerdict:
        """Execute profile against target and evaluate SLOs and baseline regression."""
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            base_url=profile.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
        )
        sem = asyncio.Semaphore(profile.concurrency)
        try:
            if profile.warmup_requests > 0:
                await asyncio.gather(
                    *[
                        self._run_one(client, sem, profile, -(i + 1))
                        for i in range(profile.warmup_requests)
                    ]
                )
            t_start = self._now()
            samples = await asyncio.gather(
                *[self._run_one(client, sem, profile, idx) for idx in range(profile.num_requests)]
            )
            duration = max(1e-6, self._now() - t_start)
        finally:
            if owns_client:
                await client.aclose()

        report = self._aggregate(profile, samples, duration)
        reasons = _evaluate_verdict(profile, report, baseline)
        return BenchmarkVerdict(report=report, passed=(len(reasons) == 0), reasons=reasons)

    def _error_sample(self, t0: float, status_code: int | None = None) -> _RequestSample:
        elapsed = (self._now() - t0) * 1000.0
        return _RequestSample(False, status_code == 429, 0.0, elapsed, [], 0, 0)

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        profile: WorkloadProfile,
        idx: int,
    ) -> _RequestSample:
        async with sem:
            payload: dict[str, object] = {
                "model": profile.model,
                "max_tokens": profile.max_tokens,
                "stream": profile.stream,
                "messages": [
                    {"role": "system", "content": profile.system_prompt},
                    {"role": "user", "content": f"{profile.user_prompt_prefix} #{idx}"},
                ],
            }
            if profile.stream:
                payload["stream_options"] = {"include_usage": True}
            headers = {"Authorization": f"Bearer {profile.api_key}"}

            t0 = self._now()
            first_token_at: float | None = None
            prev_token_at: float | None = None
            itl_samples: list[float] = []
            prompt_tokens = 0
            usage_completion_tokens: int | None = None
            chunk_count = 0

            try:
                if not profile.stream:
                    resp = await client.post("/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code >= 400:
                        return self._error_sample(t0, resp.status_code)
                    t_end = self._now()
                    usage = resp.json().get("usage", {})
                    elapsed_ms = (t_end - t0) * 1000.0
                    return _RequestSample(
                        True,
                        False,
                        elapsed_ms,
                        elapsed_ms,
                        [],
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                    )

                async with client.stream(
                    "POST", "/v1/chat/completions", headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        return self._error_sample(t0, resp.status_code)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        event = json.loads(raw)
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                            if "completion_tokens" in usage:
                                usage_completion_tokens = int(usage["completion_tokens"])
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            now = self._now()
                            if first_token_at is None:
                                first_token_at = now
                            elif prev_token_at is not None:
                                itl_samples.append((now - prev_token_at) * 1000.0)
                            prev_token_at = now
                            chunk_count += 1

                t_end = self._now()
                e2el_ms = (t_end - t0) * 1000.0
                ttft_ms = (
                    (first_token_at - t0) * 1000.0 if first_token_at is not None else e2el_ms
                )
                completion_tokens = (
                    usage_completion_tokens if usage_completion_tokens is not None else chunk_count
                )
                return _RequestSample(
                    True,
                    False,
                    ttft_ms,
                    e2el_ms,
                    itl_samples,
                    prompt_tokens,
                    completion_tokens,
                )
            except httpx.HTTPError:
                return self._error_sample(t0)

    def _aggregate(
        self,
        profile: WorkloadProfile,
        samples: list[_RequestSample],
        duration: float,
    ) -> BenchmarkReport:
        ok_samples = [s for s in samples if s.ok]
        ttfts = [s.ttft_ms for s in ok_samples]
        itls = [v for s in ok_samples for v in s.itl_ms]
        e2els = [s.e2el_ms for s in ok_samples]
        out_tokens = sum(s.completion_tokens for s in ok_samples)
        tot_tokens = sum(s.prompt_tokens + s.completion_tokens for s in ok_samples)
        out_tok_s = out_tokens / duration
        tot_tok_s = tot_tokens / duration
        goodput_count = sum(1 for s in ok_samples if s.ttft_ms <= profile.slo_p95_ttft_ms)

        return BenchmarkReport(
            profile_name=profile.name,
            model=profile.model,
            total_requests=len(samples),
            succeeded=len(ok_samples),
            failed=sum(1 for s in samples if not s.ok and not s.rate_limited),
            rate_limited_429=sum(1 for s in samples if s.rate_limited),
            duration_seconds=duration,
            requests_per_sec=len(ok_samples) / duration,
            output_tokens_per_sec=out_tok_s,
            total_tokens_per_sec=tot_tok_s,
            ttft_ms_p50=_percentile(ttfts, 50.0),
            ttft_ms_p95=_percentile(ttfts, 95.0),
            ttft_ms_p99=_percentile(ttfts, 99.0),
            itl_ms_p50=_percentile(itls, 50.0),
            itl_ms_p95=_percentile(itls, 95.0),
            e2el_ms_p50=_percentile(e2els, 50.0),
            e2el_ms_p95=_percentile(e2els, 95.0),
            goodput_rps=goodput_count / duration,
            cost_per_1m_output_usd=cost_per_million_tokens(profile.gpu_hourly_usd, out_tok_s),
            cost_per_1m_total_usd=cost_per_million_tokens(profile.gpu_hourly_usd, tot_tok_s),
        )


def _evaluate_verdict(
    profile: WorkloadProfile,
    report: BenchmarkReport,
    baseline: BenchmarkReport | None,
) -> list[str]:
    reasons: list[str] = []
    if report.succeeded == 0:
        reasons.append("Zero successful requests recorded.")
    elif report.total_requests > 0:
        success_rate = report.succeeded / report.total_requests
        if success_rate < profile.min_success_rate:
            reasons.append(
                f"Success rate {success_rate:.3f} below minimum {profile.min_success_rate:.3f}."
            )
    if report.ttft_ms_p95 > profile.slo_p95_ttft_ms:
        reasons.append(
            f"p95 TTFT {report.ttft_ms_p95:.1f}ms exceeds SLO {profile.slo_p95_ttft_ms:.1f}ms."
        )
    if profile.max_itl_p95_ms is not None and report.itl_ms_p95 > profile.max_itl_p95_ms:
        reasons.append(
            f"p95 ITL {report.itl_ms_p95:.1f}ms exceeds max {profile.max_itl_p95_ms:.1f}ms."
        )
    if baseline is not None:
        max_allowed_ttft = baseline.ttft_ms_p95 * (1.0 + profile.max_ttft_regression_pct / 100.0)
        if report.ttft_ms_p95 > max_allowed_ttft:
            reasons.append(
                f"p95 TTFT regressed to {report.ttft_ms_p95:.2f}ms "
                f"(limit {max_allowed_ttft:.2f}ms)."
            )
        min_allowed_tok_s = baseline.output_tokens_per_sec * profile.min_throughput_ratio
        if report.output_tokens_per_sec < min_allowed_tok_s:
            reasons.append(
                f"Output throughput dropped to {report.output_tokens_per_sec:.2f} tok/s "
                f"(minimum {min_allowed_tok_s:.2f} tok/s)."
            )
    return reasons
