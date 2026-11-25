from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from bench.runner import BenchmarkReport, BenchmarkRunner, WorkloadProfile


def load_baseline_report(path: Path) -> BenchmarkReport:
    """Load either a BenchmarkReport JSON or a BenchmarkVerdict JSON envelope."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "report" in raw:
        return BenchmarkReport.model_validate(raw["report"])
    return BenchmarkReport.model_validate(raw)


def main() -> int:
    """Run a workload benchmark from CLI flags and optionally gate against a baseline JSON."""
    parser = argparse.ArgumentParser(description="Multi-Tenant LLM Platform Benchmark Runner")
    parser.add_argument("--name", default="ad-hoc")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="qwen-7b")
    parser.add_argument("--num-requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument(
        "--system-prompt",
        default="You are a concise engineering assistant.",
    )
    parser.add_argument(
        "--user-prompt-prefix",
        default="Explain Kubernetes scheduling for GPU workloads",
    )
    parser.add_argument("--slo-p95-ttft-ms", type=float, default=1000.0)
    parser.add_argument("--max-itl-p95-ms", type=float, default=None)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--gpu-hourly-usd", type=float, default=0.50)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    profile = WorkloadProfile(
        name=args.name,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup_requests=args.warmup_requests,
        system_prompt=args.system_prompt,
        user_prompt_prefix=args.user_prompt_prefix,
        slo_p95_ttft_ms=args.slo_p95_ttft_ms,
        max_itl_p95_ms=args.max_itl_p95_ms,
        min_success_rate=args.min_success_rate,
        gpu_hourly_usd=args.gpu_hourly_usd,
    )
    baseline: BenchmarkReport | None = None
    if args.baseline_json is not None and args.baseline_json.exists():
        baseline = load_baseline_report(args.baseline_json)

    verdict = asyncio.run(BenchmarkRunner().run(profile, baseline=baseline))
    output = json.dumps(verdict.model_dump(), indent=2)
    print(output)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output, encoding="utf-8")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
