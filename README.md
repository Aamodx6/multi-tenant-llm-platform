# Multi-Tenant LLM Serving Platform

Kubernetes-native multi-tenant LLM serving platform with unified request control (auth, atomic Redis TPM & concurrency limits, priority admission, prefix-cache consistent hashing, SSE disconnect abort), Prometheus/DCGM/Grafana observability, guardrail-enforced self-healing, continuous benchmarking, and GitOps manifests.

## Architecture & Key Docs

- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ([Excalidraw diagram](docs/architecture.excalidraw))
- **Domain Glossary:** [GLOSSARY.md](GLOSSARY.md)
- **ADRs:** [docs/adr/](docs/adr/)
- **Research & Timed GPU Runbook:** [docs/research/end-to-end-gpu-execution-plan.md](docs/research/end-to-end-gpu-execution-plan.md)

```text
Caller ─▶ RequestControl (/gateway)
            ├─ TenantBudget (Redis Lua TPM + ZSET concurrency, conservative fallback)
            ├─ Local Priority Admission (paid vs free wait budgets, 429 Retry-After)
            ├─ CHWBL Prefix Router + PoolHealth fallback
            └─ VLLMAdapter (OpenAI SSE + include_usage + disconnect abort + /metrics poller)
         ─▶ vLLM GPU Pools (pool-8b, pool-8b-awq, pool-fallback)

Control & Delivery:
  catalog/       ModelPoolCatalog single source of truth
  controller/    SelfHealingEngine (MAD z-score + IsolationForest + K8s PDB Eviction)
  observability/ MetricsRegistry (/metrics), SLO burn-rate alerts, DCGM CSV, Grafana JSON
  deploy/        KEDA ScaledObjects, vLLM Deployments, Karpenter, Argo Rollouts, Terraform
  bench/         Concurrent SSE/HTTP BenchmarkRunner, $/1M tokens, CI regression gate
```

## Quickstart (Local Verification)

```bash
# 1. Run unit & seam tests, typechecking, and lint
pytest -v
mypy
ruff check .

# 2. Render Kubernetes, KEDA, Karpenter, Rollouts, and Observability manifests
python -m deploy.render_all --catalog catalog/pools.yaml --out-dir deploy/generated

# 3. Boot local gateway + Redis + Prometheus + Grafana (add --profile gpu on NVIDIA host)
docker compose up --build
```

## Running Benchmarks & Timed Spot-GPU Session

Follow the timed session runbook in [docs/research/end-to-end-gpu-execution-plan.md](docs/research/end-to-end-gpu-execution-plan.md):

```bash
# Record baseline / A-B arm with $/1M tokens and SLO check (p95 TTFT <= 1000ms)
python -m bench.cli \
  --name prefix-on-awq \
  --base-url http://127.0.0.1:8080 \
  --api-key sk-paid-demo \
  --model qwen-7b-awq \
  --num-requests 100 \
  --concurrency 8 \
  --gpu-hourly-usd 0.44 \
  --out-json bench/results/prefix-on-awq.json

# Gate a candidate run against a saved baseline (exits 1 on p95 TTFT or throughput regression)
python -m bench.cli \
  --name candidate \
  --api-key sk-paid-demo \
  --baseline-json bench/results/prefix-on-awq.json
```
