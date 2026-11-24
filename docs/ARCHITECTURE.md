# Multi-Tenant LLM Serving Platform — Architecture

![Multi-Tenant LLM Serving Platform Architecture](./architecture.png)

Editable Excalidraw source: [architecture.excalidraw](./architecture.excalidraw) · Domain language: [GLOSSARY.md](../GLOSSARY.md) · Decisions: [docs/adr/](./adr/)

---

## 1. Overview

A Kubernetes-based multi-tenant LLM serving platform designed around **deep modules with explicit seams** so every core behavior—token budgeting, priority admission, prefix-cache routing, SSE disconnect cancellation, self-healing guardrails, and benchmark gating—runs and tests deterministically before and during GPU cluster execution.

| Plane | Modules & Files | Responsibility |
| :--- | :--- | :--- |
| **Catalog (Source of Truth)** | [`catalog/pool_catalog.py`](../catalog/pool_catalog.py), [`catalog/pools.yaml`](../catalog/pools.yaml) | Validates pool invariants, fallback DAG, isolation modes (`shared`, `dedicated`, `fallback`), and renders downstream configs for Request Control, KEDA, Rollouts, NVMe prefetch, and `vllm` CLI args. |
| **Request Control** | [`gateway/request_control.py`](../gateway/request_control.py), [`gateway/server.py`](../gateway/server.py) | Single Starlette/ASGI module uniting Bearer API key auth, `TenantBudget`, local priority admission, CHWBL prefix routing, `PoolHealth` fallback, SSE streaming, and disconnect abort. |
| **Tenant Budget & Pool Health** | [`gateway/tenant_budget.py`](../gateway/tenant_budget.py), [`gateway/pool_health.py`](../gateway/pool_health.py) | Atomic Redis Lua (`HSET` + `EXPIRE` + `redis.call('TIME')`) TPM and `ZSET` concurrency permits with replica-scaled conservative fallback on Redis outage; cross-pod TTL-bounded `PoolHealth` overrides with probe hysteresis. |
| **AIOps Self-Healing** | [`controller/self_healing.py`](../controller/self_healing.py) | Pure `SelfHealingEngine` combining Iglewicz–Hoaglin MAD Modified Z-Score + `IsolationForest` scoring with guardrails (`dry_run`, max-1-node drain, cooldowns, 2-failure paging) behind a `ClusterActuator` seam. |
| **Observability** | [`observability/metrics.py`](../observability/metrics.py), [`observability/bundle.py`](../observability/bundle.py) | Isolated Prometheus `MetricsRegistry` (`/metrics`), periodic `VLLMAdapter.scrape_metrics()` lifespan loop, multi-window burn-rate alerts, pinned DCGM `counters.csv` DaemonSet, and Grafana dashboard JSON. |
| **Benchmarking** | [`bench/runner.py`](../bench/runner.py), [`bench/cli.py`](../bench/cli.py) | Concurrent OpenAI HTTP/SSE load driver measuring p50/p95/p99 TTFT/ITL/E2EL, `goodput_rps`, output & blended `$/1M tokens`, typed `ArmComparison`, and CI baseline regression gating. |
| **Delivery & GitOps** | [`deploy/manifest_renderer.py`](../deploy/manifest_renderer.py), [`deploy/terraform/`](../deploy/terraform/main.tf), [`docker-compose.yml`](../docker-compose.yml) | Renders KEDA `ScaledObject`s, vLLM `Deployment`s, `PodDisruptionBudget`s, `NetworkPolicy`, Karpenter `NodePool`/`EC2NodeClass`, Argo `Service`/`Rollout`/`AnalysisTemplate`, plus Terraform EKS/S3 and local Docker Compose. |

---

## 2. Request Lifecycle (`RequestControl`)

1. **Schema & Alias Resolution:** Validates the OpenAI `/v1/chat/completions` payload (`ConfigDict(extra="allow")`, returning OpenAI-formatted `400 invalid_request_error` on malformed JSON/schemas) and maps model aliases via `ModelPoolCatalog.resolve_alias`.
2. **Authentication & `Permit` Reservation (`TenantBudget`):** Hashes the Bearer `API Key` with SHA-256, looks up `TenantRecord` (Redis cache backed by Postgres source), and atomically reserves `est_prompt + max_tokens` (UTF-8 byte-ratio + ChatML estimate) and a `ZSET` concurrency slot via `_RESERVE_LUA`. On Redis outage, degrades to replica-scaled local buckets.
3. **In-Process Priority Admission:** Local priority heap (`paid` before `free`) checks expected queue wait against tier budgets, shedding excess `free` traffic with `429` + `Retry-After` and cancelling safely without leaking `Permit`s or `_in_flight` slots on disconnect.
4. **CHWBL Prefix Routing & `PoolHealth` Fallback:** Selects a `HEALTHY` pool (falling back cleanly along the validated `fallback_pool` DAG when `PoolHealth` reports `DEGRADED`/`UNHEALTHY`). Within the pool, `_pick_replica` uses a 64-vnode consistent hash ring (`sha256` of system prompt prefix) with bounded-load spillover (`ceil(1.25 * avg_load)` and KEDA queue/KV caps).
5. **Streaming, Disconnect Abort & Usage Reconciliation:** `VLLMAdapter` requests `stream_options: {"include_usage": True}`, passes OpenAI SSE chunks cleanly (`role`, `delta`, `finish_reason="stop"`, trailing `usage`), and on caller `http.disconnect` closes the upstream `httpx` stream inside `anyio.CancelScope(shield=True)` so vLLM aborts KV generation. Unused reserved tokens and concurrency slots are reconciled on completion or abort.

---

## 3. Seams & Adapters (Testability & Portability)

| Seam | Production Adapter | Deterministic Test Adapter | Purpose |
| :--- | :--- | :--- | :--- |
| `InferenceEngine` | `VLLMAdapter` (`httpx` + `/metrics`) | `FakeEngineAdapter` | Decouples `RequestControl` HTTP/SSE/routing tests from live GPU pods. |
| `BudgetAdapter` | `RedisPostgresAdapter` (`Lua` + `TIME`) | `InMemoryBudgetAdapter` (`fakeredis` also tested) | Encapsulates Redis failure degradation and atomic bucket math. |
| `PoolHealth` | `RedisPoolHealthAdapter` (atomic Lua) | `PoolHealthStore` | Shares slow-path `DEGRADED` TTL overrides across pods without ArgoCD drift. |
| `ClusterActuator` | `K8sActuatorAdapter` (`policy/v1` Eviction + `429` retry) | `SimulatedActuatorAdapter` | Tests full anomaly-to-drain/restart/escalation guardrails in milliseconds. |

---

## 4. AIOps Control Loop & Guardrails (`SelfHealingEngine`)

- **Univariate + Multivariate Scoring:** Computes Iglewicz–Hoaglin MAD Modified Z-Score on `ttft_ms_series` (immune to jitter and baseline outliers) and optional `IsolationForest` scoring on `TelemetryFeature(ttft_ms, kv_cache_usage, gpu_temp_c)`.
- **Signal → Action Mapping:**
  - `xid_errors > 0`, `ecc_errors > 0`, or `gpu_temp_c >= 85.0` → `CORDON_AND_DRAIN` via `ClusterActuator` (cordon + `policy/v1` Eviction respecting PDBs, skipping DaemonSets).
  - `kv_cache_usage >= 0.95` or `oom_restarts > 0` (or joint anomaly) → `MARK_POOL_DEGRADED` via `PoolHealth` TTL override + `RESTART_POD` (without mutating ArgoCD-managed Deployment args).
- **Guardrails:** `dry_run` mode, action deduplication, per-target cooldowns, at most 1 concurrent draining node (`complete_drain()` to release), structured `AuditEntry` log, and `ESCALATE_TO_HUMAN` (`page_human`) after 2 consecutive failures.

---

## 5. Observability, Delivery & Benchmarking

- **Metrics & Alerts:** `RequestControl` exposes `GET /metrics` (9 Prometheus metrics covering requests, 429s, queue wait, streaming TTFT, E2EL, fallbacks, tokens, anomaly scores, and remediation actions) and polls vLLM `/metrics` in its ASGI lifespan. `render_observability_bundle` generates multi-window fast/slow burn-rate alerts, explicit DCGM `counters.csv` + pinned `DaemonSet`, and Grafana dashboard JSON.
- **Deploy & GitOps:** `render_deploy_manifests` emits KEDA `ScaledObject`s (`vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `activationThreshold` for scale-to-zero), vLLM `Deployment`s (`/dev/shm`, NVMe cache mount, startup/readiness probes, `preStop` drain), `NetworkPolicy`, Karpenter `g6`/`g5` spot+on-demand `NodePool`/`EC2NodeClass`, and Argo `Rollout` + `AnalysisTemplate`.
- **Execution Status & Roadmap:** Local CPU Docker Compose (`request-control`, `redis`, `prometheus`, `grafana`) and 32 unit/seam tests pass locally. GPU benchmarking (`bench/cli.py`), live spot-GPU A/B runs, NVMe prefetch `DaemonSet`, and full EKS/Karpenter deployment follow the runbook in [end-to-end-gpu-execution-plan.md](./research/end-to-end-gpu-execution-plan.md) (hardening items tracked in [present-architecture-primary-review.md](./research/present-architecture-primary-review.md)).

---

## 6. Architecture Decisions (ADRs)

1. [0001: Collapse Gateway, Admission, and Router into Request Control](./adr/0001-request-control-single-module.md)
2. [0002: Separate Fast-Path Routing from Slow-Path Remediation via Pool Health](./adr/0002-pool-health-ownership.md)
3. [0003: Encapsulate Redis Outage Degradation in Tenant Budget and Keep Queues Local](./adr/0003-tenant-budget-and-local-queues.md)
4. [0004: Co-Locate Anomaly Scoring and Remediation Policy Behind Cluster Actuator](./adr/0004-self-healing-actuator-seam.md)
5. [0005: Centralize Model Pool Configuration in Model Pool Catalog](./adr/0005-model-pool-catalog.md)
