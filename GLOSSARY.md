# Multi-Tenant LLM Serving Platform

Shared domain language for multi-tenant request admission, GPU model pool routing, token budgeting, and self-healing operations.

## Language

### Tenancy & Admission

**Tenant**:
An organization or workload account that holds a priority class, token-per-minute quota, and one or more API keys.
_Avoid_: Client, user, account

**API Key**:
A hashed credential issued to a tenant and presented on each request to authenticate and resolve tenant limits.
_Avoid_: Token, secret

**Caller**:
An external application or load generator sending OpenAI-compatible HTTP requests to the platform.
_Avoid_: Tenant, client

**Priority Class**:
A tenant service tier (`paid` or `free`) that determines admission queue ordering, queue wait budgets, and load-shedding eligibility.
_Avoid_: Plan, role

**Permit**:
A bounded token reservation issued for a single request against a tenant's budget and reconciled with actual usage when generation ends.
_Avoid_: Lease, ticket, quota check

### Serving & Routing

**Model Pool**:
A homogeneous group of inference replicas serving one model and quantization behind a shared health status and autoscaling policy.
_Avoid_: Deployment, cluster, tenant pool

**Request Control**:
The unified edge runtime that authenticates callers, enforces tenant budgets, queues requests by priority, routes by prompt prefix, and streams responses.
_Avoid_: Gateway-router split, proxy layer

**Pool Health**:
The operational state (`healthy`, `degraded`, or `unhealthy`) of a model pool used to drive fallback routing without mutating deployment manifests.
_Avoid_: Circuit breaker state, readiness flag

**LoRA Adapter**:
A tenant-owned or shared low-rank weight delta served dynamically over a compatible base model pool (`--enable-lora`) and authorized per tenant before queue admission.
_Avoid_: Fine-tuned model, plugin, sidecar weights

**Cache Salt**:
A gateway-derived cryptographic per-tenant token injected into vLLM's first KV-block hash (`cache_salt`) to isolate automatic prefix cache namespaces against cross-tenant timing side channels.
_Avoid_: Cache key, API key hash, session ID

### Metering & Delivery

**Usage Event**:
An immutable post-completion or post-abort token accounting record (`trace_id`, `tenant_id`, `pool`, `prompt_tokens`, `completion_tokens`, optional `lora_name`) appended to Redis Streams for idempotent hourly Postgres rollups.
_Avoid_: Metric sample, invoice line, log entry

**Weight Prefetch DaemonSet**:
A GPU-node Kubernetes `DaemonSet` that syncs model and LoRA weights from S3 to local RAID0 NVMe (`hostPath`) via `s5cmd` with atomic `.ready` publication before inference pods start.
_Avoid_: Model downloader job, sidecar cache

### Self-Healing & Configuration

**Telemetry Window**:
A time-ordered snapshot of GPU hardware and inference engine metrics evaluated to detect anomalies.
_Avoid_: Scrape payload, alert

**Remediation Plan**:
An ordered, guardrail-checked set of cluster or pool-health actions emitted in response to an unhealthy telemetry window.
_Avoid_: Runbook script, alert rule

**Model Pool Catalog**:
The validated single source of truth defining all model pools, aliases, fallback edges, scaling bounds, and SLO targets.
_Avoid_: Helm values, config map
