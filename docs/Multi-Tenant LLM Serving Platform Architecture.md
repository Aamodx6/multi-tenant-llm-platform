# Multi-Tenant LLM Serving Platform: Architecture

Architecture blueprint, November 24, 2026

## Overview

A Kubernetes-based platform that serves open LLMs to multiple tenants with GPU-aware autoscaling, full observability, automated remediation, continuous benchmarking and GitOps delivery. It is designed to cover three roles at once: AI infrastructure, AIOps and GPU inference engineering.

## Layers at a glance

| Layer | Components | Purpose |
| --- | --- | --- |
| Clients | Tenant apps using an OpenAI-style API with SSE streaming | Entry point |
| Request path | Gateway, admission control, router | Auth, token limits, priority, routing |
| Inference plane | vLLM pool A (8B, AWQ or FP8), vLLM pool B (70B, tensor parallel), fallback pool (small model, spot GPU) | GPU compute on Kubernetes node pools |
| State and storage | Model store (S3 or MinIO), node cache (NVMe weights), Redis, Postgres | Weights, limits, router state, tenants, usage |
| Observability | Prometheus and DCGM, Loki and traces, Grafana and burn-rate alerts | Metrics, logs, traces, SLOs |
| Control loops (AIOps) | KEDA and Karpenter, anomaly detector, remediation loop | Scaling and self-healing |
| Delivery (GitOps) | CI and bench gate, ArgoCD and Rollouts, Terraform and model registry | Safe, repeatable releases |

Data flow: clients, gateway, admission control, router, inference pools. Telemetry flows from the inference plane into observability, then into the control loops, whose actions return to the cluster. Delivery deploys into the cluster.

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
| Gateway | FastAPI or Envoy with ext-authz | Stateless, HPA on CPU and RPS, per-tenant metrics labels |
| Admission | In-process queues backed by Redis | Priority classes, load shedding, per-tenant concurrency caps |
| Router | Gateway module | Prefix hashing, bounded load, health from engine metrics, fallback chain |
| Inference pool | One Deployment per model and quantization | 1 GPU per pod (or 2 to 4 for tensor parallel, same node), `/dev/shm` sized, startup probe covers model load, readiness after a warmup request |
| Weight delivery | S3 or MinIO, DaemonSet prefetch to node NVMe | Cuts cold start from minutes to seconds on warm nodes |
| Node pools | Small GPU (L4 or A10G), large GPU (A100 or H100), CPU | Taints on GPU pools, nodeSelectors per pool |
| Redis | Single primary with replica | Token buckets, API key cache, router state; losing it must degrade to conservative limits, not an outage |
| Postgres | Managed or CloudNativePG | Tenants, keys, quotas, model aliases, usage rollups |

## Autoscaling

- **Pod level (KEDA with Prometheus triggers):** scale on `vllm:num_requests_waiting` per replica and KV-cache usage above about 80%. GPU utilization is a secondary signal only, since it saturates early on LLM workloads.
- **Node level (Karpenter or Cluster Autoscaler):** spot plus on-demand mix, with low-priority pause pods as headroom so new nodes are not on the critical path.
- **Scale-down:** long cooldown, `preStop` drain, and a `terminationGracePeriodSeconds` longer than the longest allowed generation.
- **Minimums:** at least one replica for primary pools. Scale-to-zero only for the fallback pool or idle tenants.

## Observability

- **Engine metrics:** TTFT, time per output token, end-to-end latency, running and waiting requests, KV-cache usage, prefix-cache hit rate.
- **GPU metrics (DCGM):** utilization, framebuffer used, power, temperature, XID and ECC errors, NVLink.
- **Gateway metrics:** tokens per tenant, `429` rate, queue wait, router fallback rate.
- **SLOs:** for example p95 TTFT under 1s for the premium class and 99.5% availability, with error budget burn-rate alerts at two windows (fast and slow).
- **Privacy:** prompts and completions are not logged by default. Tracing carries IDs and token counts only.

## AIOps control loops

The anomaly detector reads DCGM and engine time series, computes per-GPU rolling features, and exposes an anomaly score as a Prometheus metric. Alerts feed the remediation loop, a small operator driven by a `RemediationPolicy` CRD.

| Signal | Action |
| --- | --- |
| XID or ECC errors, thermal throttling | Cordon and drain the node, reschedule pods |
| KV cache above about 95%, OOM restarts | Lower `max-num-seqs`, then restart the pod |
| Queue depth and p95 TTFT rising | Scale out, shed lowest priority class |
| Pool failing health checks | Router shifts traffic to fallback pool |
| Canary breaches SLO | Argo Rollouts automatic rollback |

**Guardrails:** dry-run mode, at most one node drained at a time, rate limits per action, an audit log, and escalation to a human page when an action fails twice.

## Delivery

- **CI:** lint, unit tests, image build, Trivy scan, cosign signing, then a benchmark job on an ephemeral GPU runner that compares against a stored baseline and fails on a p95 or throughput regression.
- **CD:** ArgoCD ApplicationSets for dev, stage and prod. Argo Rollouts canary at 5%, 25% and 100% with an AnalysisTemplate on TTFT and error rate.
- **Infra:** Terraform modules for network, cluster, node pools, buckets and IAM. Model versions tracked in a registry (MLflow or a versioned bucket layout).

## Security and isolation

- NetworkPolicies: only the router can reach inference pods, and inference pods have no general egress.
- Hashed API keys, secrets via External Secrets, non-root signed images.
- Per-tenant concurrency and token caps prevent noisy neighbors. Separate pools for tenants that need hard isolation.

## Key decisions (ADRs)

1. vLLM as the primary engine, with SGLang or TensorRT-LLM kept behind the same pool interface for benchmarking.
2. Custom gateway instead of LiteLLM, because token-aware limits and prefix routing are the point of the project.
3. KEDA on queue depth instead of CPU or GPU utilization.
4. Prefix-hash routing with bounded load instead of plain round-robin.
5. Remediation as a policy-driven operator with guardrails, not ad hoc scripts.

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
