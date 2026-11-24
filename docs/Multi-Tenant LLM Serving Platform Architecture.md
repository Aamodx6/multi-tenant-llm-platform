# Multi-Tenant LLM Serving Platform: Architecture

Architecture blueprint, November 24, 2026

## Overview

A Kubernetes-based platform that serves open LLMs to multiple tenants with GPU-aware autoscaling, full observability, automated remediation, continuous benchmarking and GitOps delivery. It is designed to cover three roles at once: AI infrastructure, AIOps and GPU inference engineering.

## Layers at a glance

| Layer | Components | Purpose |
| --- | --- | --- |
| Callers | Tenant apps using an OpenAI-style HTTP interface with SSE streaming | Entry point |
| Request path | Unified Request Control (`gateway`: auth, `TenantBudget`, local admission, prefix router) | Auth, token limits, priority, routing |
| Inference plane | vLLM pool A (8B, AWQ or FP8), vLLM pool B (70B, tensor parallel), fallback pool (small model, spot GPU) | GPU compute on Kubernetes node pools |
| State and storage | `ModelPoolCatalog` (single source of truth), Model store (S3 or MinIO), node cache (NVMe weights), Redis (`TenantBudget` + `PoolHealth`), Postgres | Pool config, weights, limits, tenants, usage |
| Observability | Prometheus and DCGM, Loki and traces, Grafana and burn-rate alerts | Metrics, logs, traces, SLOs |
| Control loops (AIOps) | KEDA and Karpenter, unified `SelfHealingEngine` (`TelemetryWindow` scoring + `RemediationPlan` via `ClusterActuator`) | Scaling and self-healing |
| Delivery (GitOps) | CI and bench gate, ArgoCD and Rollouts, Terraform and model registry | Safe, repeatable releases |

Data flow: callers, `RequestControl`, inference pools. `ModelPoolCatalog` drives pool configuration across routing, KEDA, and rollouts. Telemetry flows from the inference plane into observability and `SelfHealingEngine`, which coordinates slow-path actions via `ClusterActuator` and `PoolHealth` overrides. Delivery deploys into the cluster.

## Request lifecycle

1. **Gateway:** TLS, API key lookup (hashed key, cached in Redis, Postgres as source of truth), tenant resolved, model alias mapped to a pool.
2. **Token budget:** count prompt tokens, reserve `max_tokens` against the tenant's token-per-minute bucket (atomic Lua script in Redis), reconcile with actual usage after the response.
3. **Admission:** priority queues per tenant class. If expected queue wait exceeds the class budget, return `429` with `Retry-After` instead of queuing forever.
4. **Router:** hash the prompt prefix (system prompt plus first N tokens) to pick a replica, with a bounded-load rule so a hot prefix cannot overload one pod. Fall back to least-loaded using `num_requests_waiting` and KV-cache usage. If a pool is unhealthy or saturated, route to the fallback pool.
5. **Streaming:** SSE passthrough. A client disconnect aborts the request in vLLM so GPU time is not wasted.
6. **Metering:** token counts go to a usage stream (Redis Streams, or Kafka later), then are rolled up into Postgres. Every request carries one trace ID from gateway to engine.

## Component specs

| Component | Implementation | Key design points |
| --- | --- | --- |
| Request Control | Single Python (`Starlette` / `FastAPI` + `Pydantic`) module | Combines auth, `TenantBudget`, local priority admission, prefix-aware routing, and SSE streaming behind an `InferenceEngine` seam (`vLLMAdapter` / `FakeEngineAdapter`) |
| Admission | In-process priority queues per replica | Local priority classes, wait-budget load shedding (`429 Retry-After`), no Redis dependency on queue state |
| Tenant Budget | `TenantBudget` module (`RedisPostgresAdapter` / `InMemoryBudgetAdapter`) | Hashed API key lookup, atomic Redis token-bucket reserve/reconcile, conservative local fallback encapsulated on Redis outage |
| Pool Health | `PoolHealth` seam (`healthy`, `degraded`, `unhealthy`) | Fast-path routing reads state; slow-path remediation sets TTL-bounded `degraded` overrides without fighting ArgoCD |
| Model Pool Catalog | `ModelPoolCatalog` schema and renderer | Single validated source of truth for pool aliases, fallback chains, isolation mode (`shared` vs `dedicated`), KEDA bounds, and SLO targets |
| Inference pool | One Deployment per model pool and quantization | 1 GPU per pod (or 2 to 4 for tensor parallel, same node), `/dev/shm` sized, startup probe covers model load, readiness after a warmup request |
| Weight delivery | S3 or MinIO, DaemonSet prefetch to node NVMe | Cuts cold start from minutes to seconds on warm nodes |
| Node pools | Small GPU (L4 or A10G), large GPU (A100 or H100), CPU | Taints on GPU pools, nodeSelectors per pool |
| Redis | Single primary with replica | Encapsulated inside `TenantBudget` and `PoolHealth`; losing it degrades cleanly to conservative local limits |
| Postgres | Managed or CloudNativePG | Tenants, keys, quotas, model aliases, usage rollups |

## Autoscaling

- **Pod level (KEDA with Prometheus triggers):** scale on `vllm:num_requests_waiting` per replica and KV-cache usage above about 80%. GPU utilization is a secondary signal only, since it saturates early on LLM workloads.
- **Node level (Karpenter or Cluster Autoscaler):** spot plus on-demand mix, with low-priority pause pods as headroom so new nodes are not on the critical path.
- **Scale-down:** long cooldown, `preStop` drain, and a `terminationGracePeriodSeconds` longer than the longest allowed generation.
- **Minimums:** at least one replica for primary shared pools. Scale-to-zero only for the fallback pool or `dedicated` isolation pools declared in `ModelPoolCatalog`.

## Observability

- **Engine metrics:** TTFT, time per output token, end-to-end latency, running and waiting requests, KV-cache usage, prefix-cache hit rate.
- **GPU metrics (DCGM):** utilization, framebuffer used, power, temperature, XID and ECC errors, NVLink.
- **Gateway metrics:** tokens per tenant, `429` rate, queue wait, router fallback rate.
- **SLOs:** for example p95 TTFT under 1s for the premium class and 99.5% availability, with error budget burn-rate alerts at two windows (fast and slow).
- **Privacy:** prompts and completions are not logged by default. Tracing carries IDs and token counts only.

## AIOps control loops

`SelfHealingEngine` co-locates rolling `TelemetryWindow` anomaly scoring (z-score and threshold baseline) with `RemediationPolicy` guardrail evaluation, emitting a deterministic `RemediationPlan` across a `ClusterActuator` seam (`K8sActuatorAdapter` in prod, `SimulatedActuatorAdapter` in tests). Anomaly scores are still exported as Prometheus metrics for Grafana without adding a Prometheus/Alertmanager hop to the critical remediation path.

| Signal | Action |
| --- | --- |
| XID or ECC errors, thermal throttling | Cordon and drain the node, reschedule pods |
| KV cache above about 95%, OOM restarts | Mark pool `degraded` via `PoolHealth` (shifting new/excess traffic without mutating ArgoCD-managed Deployment args), then restart the unhealthy pod |
| Queue depth and p95 TTFT rising | `RequestControl` sheds lowest priority class on the fast path; KEDA scales out on the slow path |
| Pool failing health checks | `RequestControl` shifts traffic to fallback pool (`PoolHealth` hysteresis prevents flapping) |
| Canary breaches SLO | Argo Rollouts automatic rollback |

**Guardrails:** dry-run mode, at most one node drained at a time, rate limits per action, an audit log, and escalation to a human page when an action fails twice.

## Delivery

- **CI:** lint, unit tests, image build, Trivy scan, cosign signing, then a benchmark job on an ephemeral GPU runner that compares against a stored baseline and fails on a p95 or throughput regression.
- **CD:** ArgoCD ApplicationSets for dev, stage and prod. Argo Rollouts canary at 5%, 25% and 100% with an AnalysisTemplate on TTFT and error rate.
- **Infra:** Terraform modules for network, cluster, node pools, buckets and IAM. Model versions tracked in a registry (MLflow or a versioned bucket layout).

## Security and isolation

- NetworkPolicies: only `RequestControl`, Prometheus scrapes, and kubelet probes can reach inference pods, and inference pods have no general egress beyond telemetry export and model store cache fallback.
- Hashed API keys, secrets via External Secrets, non-root signed images.
- Per-tenant concurrency and token caps prevent noisy neighbors. Separate pools for tenants that need hard isolation.

## Key decisions (ADRs)

Full records live in `docs/adr/` and domain terms in [GLOSSARY.md](../GLOSSARY.md):

1. [0001: Collapse Gateway, Admission, and Router into Request Control](./adr/0001-request-control-single-module.md)
2. [0002: Separate Fast-Path Routing from Slow-Path Remediation via Pool Health](./adr/0002-pool-health-ownership.md)
3. [0003: Encapsulate Redis Outage Degradation in Tenant Budget and Keep Queues Local](./adr/0003-tenant-budget-and-local-queues.md)
4. [0004: Co-Locate Anomaly Scoring and Remediation Policy Behind Cluster Actuator](./adr/0004-self-healing-actuator-seam.md)
5. [0005: Centralize Model Pool Configuration in Model Pool Catalog](./adr/0005-model-pool-catalog.md)

### ADR template

| Field | Content |
| --- | --- |
| Title | Short decision name |
| Status | Proposed, accepted or superseded |
| Context | The problem and constraints |
| Decision | What was chosen |
| Alternatives | Options considered and why they lost |
| Consequences | Trade-offs, risks and follow-ups |
| Evidence | Benchmark or test result that supports it |
