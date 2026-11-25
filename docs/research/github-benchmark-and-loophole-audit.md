# GitHub Benchmark & Security / Correctness Loophole Audit

**Repository:** [`Aamodx6/multi-tenant-llm-platform`](https://github.com/Aamodx6/multi-tenant-llm-platform) (audited at commit `8ced565`)  
**Baseline Checks Observed:** `pytest -q` (`43 passed in 28.88s`), `ruff check .` (clean), `mypy` (`28 source files` clean, with unused override note in `pyproject.toml`).  
**Scope:** Primary-source benchmark against production open-source multi-tenant LLM serving stacks on GitHub (`vllm-project/aibrix`, `vllm-project/production-stack`, `kubernetes-sigs/gateway-api-inference-extension`, `llm-d/llm-d`, `BerriAI/litellm`, `envoyproxy/ai-gateway`), combined with a line-level security, quota/billing, concurrency, Kubernetes/Terraform, and repo hygiene audit of the current codebase.

---

## 1. Executive Summary & OSS GitHub Benchmark Matrix

`Aamodx6/multi-tenant-llm-platform` implements a clean, modular Python gateway (`RequestControl`), Redis-backed token/concurrency enforcement (`TenantBudget`), Redis-backed pool health state (`PoolHealth`), Redis Streams-to-Postgres billing rollup (`BillingRollupWorker`), MAD/IsolationForest self-healing (`SelfHealingEngine`), catalog-driven Kubernetes manifest generation, and an async OpenAI/SSE benchmark suite.

However, benchmarking against mature OSS control planes and auditing the current code reveals **critical quota/billing bypass loopholes, async event-loop blocking, single-node/EKS deployment blockers, and missing production guardrails** that are already solved in comparable GitHub projects.

| Capability | This Repo (`Aamodx6/multi-tenant-llm-platform`) | [`vllm-project/aibrix`](https://github.com/vllm-project/aibrix) | [`vllm-project/production-stack`](https://github.com/vllm-project/production-stack) | [`kubernetes-sigs/gateway-api-inference-extension`](https://github.com/kubernetes-sigs/gateway-api-inference-extension) + [`llm-d/llm-d`](https://github.com/llm-d/llm-d) | [`BerriAI/litellm`](https://github.com/BerriAI/litellm) / [`envoyproxy/ai-gateway`](https://github.com/envoyproxy/ai-gateway) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefix-Cache Routing** | CHWBL over 64 vnodes hashing first 256 chars of first `system` (or first) message (`gateway/request_control.py#L1039-L1116`). | Global/local prefix-cache indexers + composable soft-scoring (`prefix-cache` + `least-request`), Redis cross-gateway sync. | `vllm_router` KV/prefix-aware routing + **LMCache** CPU/disk/remote KV offloading. | EPP `prefix-cache-scorer` + `queue-scorer` + `kv-cache-utilization-scorer`; `llm-d` event-driven KV-cache indexer & P2P KV transfer. | Envoy AI Gateway uses EPP `ext-proc`; LiteLLM focuses on model/RPM/TPM routing & fallback. |
| **Multi-Tenant Cache Isolation** | Strips client `cache_salt`, injects per-tenant HMAC-SHA256 `cache_salt` (`gateway/request_control.py#L598-L612`). **Broken in Compose** (`v0.7.3` ignores `cache_salt`). | Dedicated per-tenant isolation or gateway-injected isolation policies + Envoy Gateway. | Helm values configure vLLM engine flags; relies on gateway/vLLM `cache_salt` (`>=0.9.0`). | `InferencePool` + per-model/tenant routing & flow-control fairness policies. | Envoy AI Gateway header/identity policies; vLLM `cache_salt` (`CVE-2025-46570` fix in `v0.9.0`). |
| **Dynamic LoRA Serving** | Catalog schema + `VLLMAdapter.ensure_lora_loaded` per-gateway-pod set; **no adapters in `catalog/pools.yaml`**, no unload/LRU sync, no LoRA-aware capacity scoring. | `ModelAdapter` CRD + `aibrix-runtime` sidecar + `model-route-controller` packing/evicting LoRAs across shared base pods. | Helm `enableLoRA` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. | EPP capability tracking routes directly to replicas with target LoRA already resident in VRAM. | LiteLLM routes by model/deployment alias; Envoy routes via `AIGatewayRoute` headers. |
| **Token Quota & Billing** | Redis Lua token bucket + concurrency ZSET + Redis Streams -> Postgres hourly rollup. **Bypassable via `max_completion_tokens`, `n`, `tools`, debt clamping (`max(0, ...)`).** | Per-model RPS rate limiting in Envoy Gateway data plane. | Relies on upstream API gateway for auth/billing. | Priority & fairness flow control (`sheddable` vs `critical` workloads). | LiteLLM: virtual keys, Redis Lua TPM/RPM + DB spend; Envoy AI Gateway: `llmRequestCosts` (`InputToken`, `CachedInputToken`, `OutputToken`, `CEL`) + Redis global rate limit. |
| **Autoscaling & GPU Ops** | KEDA `ScaledObject` (deadlocks at `minReplicaCount: 0`), Karpenter `NodePool`, Argo Rollout (missing `trafficRouting`), unwired `SelfHealingEngine`. | Custom LLM Autoscaler (APA) on queue/KV/throughput + `ModelClaim`/`ModelWarmup`. | Prometheus + HPA/KEDA + Helm stack. | Gateway API `InferencePool` + HPA/KEDA on engine & gateway metrics. | Standard K8s HPA / Envoy Gateway scaling. |
| **CI/CD & Packaging** | Local `pytest`/`ruff`/`mypy` pass, but **no `.github/workflows/`**, unpinned `:latest` image, missing `psycopg` DB driver. | Full Go/Python CI, Helm charts, CRDs, e2e suites. | Helm charts, tutorials, router unit/integration tests, CI workflows. | Conformance suites, CRD validation, Go CI. | GitHub Actions CI, Docker images, security advisories/CVE process. |

---

## 2. Critical Security & Quota / Billing Loopholes

### 2.1 `max_completion_tokens`, `n`, and `tools` Parameter Smuggling Bypass TPM Budget
* **Severity:** **Critical**
* **Evidence:**
  * [`gateway/request_control.py#L53-L62`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L53-L62): `ChatCompletionRequest` sets `model_config = ConfigDict(extra="allow")` and validates only `max_tokens: int = Field(default=256, ge=1, le=32768)`.
  * [`gateway/request_control.py#L44-L50`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L44-L50): `ChatMessage` sets `model_config = ConfigDict(extra="allow")` with `content: str`.
  * [`gateway/request_control.py#L478-L485`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L478-L485): `_estimate_prompt_tokens` only reads `msg.content` and ignores `tools`, `tool_calls`, `response_format`, `name`, and `extra` fields.
  * [`gateway/request_control.py#L606-L612`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L606-L612) & [`#L694-L701`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L694-L701): `est_total = est_prompt + req.max_tokens` is reserved, while `req.model_dump()` forwards all `extra="allow"` fields verbatim to vLLM.
* **Failure / Abuse Scenario:**
  1. A tenant omits `max_tokens` (defaulting reservation to `256`) and supplies `"max_completion_tokens": 16384` (which OpenAI and vLLM prioritize over `max_tokens`) or `"n": 16` (multiplying completion generation by 16x), consuming far more GPU tokens than reserved.
  2. A tenant passes large JSON schemas in `tools` or `response_format` with a 1-word `content` (`est_prompt ≈ 5` tokens), bypassing pre-call TPM admission.
  3. Forwarding arbitrary `extra="allow"` fields also permits passing internal/expensive vLLM parameters (`best_of`, `prompt_logprobs`, `guided_json`, `chat_template`, etc.).
* **OSS Benchmark Reference:** [vLLM OpenAI Protocol](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) supports `max_completion_tokens` and `n`; [Envoy AI Gateway `AIGatewayRoute`](https://aigateway.envoyproxy.io/) and LiteLLM explicitly account for full request token cost and restrict/validate parameter schemas rather than forwarding unvalidated `extra="allow"` blobs.
* **Remediation:**
  1. Explicitly declare `max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)` and `n: int = Field(default=1, ge=1, le=1)` (or multiply reservation by `n` and parse all `choices`), compute `effective_max_tokens = min(pool_spec.max_model_len, req.max_completion_tokens or req.max_tokens)`, and reserve `est_prompt + (effective_max_tokens * req.n)`.
  2. Replace `ConfigDict(extra="allow")` on `ChatCompletionRequest` with an explicit allowlist of safe OpenAI sampling fields (`temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`, `seed`, `user`, `tools`, `tool_choice`, `response_format`) and include serialized `tools` / `tool_calls` / `response_format` / multimodal content parts in `_estimate_prompt_tokens`.

---

### 2.2 Budget Debt Forgiveness (`max(0.0, ...)`) Enables Unbounded Quota Overdraft
* **Severity:** **Critical**
* **Evidence:**
  * [`gateway/tenant_budget.py#L143`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L143) (`InMemoryBudgetAdapter.reconcile`): `bucket.tokens = min(cap, max(0.0, bucket.tokens + delta))`
  * [`gateway/tenant_budget.py#L223`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L223) (`_RECONCILE_LUA`): `tokens = math.min(cap, math.max(0.0, tokens + delta))`
* **Failure / Abuse Scenario:** When `actual_tokens > permit.reserved_tokens` (e.g., due to `max_completion_tokens`, `tools`, non-ASCII BPE expansion, or tool-call output), `delta` is negative. If `tokens + delta < 0`, both Python and Redis Lua **clamp the bucket to `0.0`**, silently erasing all excess token debt. The bucket immediately begins refilling from `0` at `tpm_limit / 60` tokens/sec, allowing repeated multi-fold TPM quota overruns.
* **OSS Benchmark Reference:** Production token-bucket/cost limiters (e.g., [LiteLLM](https://github.com/BerriAI/litellm) and [Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway)) allow negative bucket balance (token debt) so subsequent requests wait until refill repays the actual overage.
* **Remediation:** Remove the `0.0` floor on negative `delta` in both `InMemoryBudgetAdapter.reconcile` and `_RECONCILE_LUA`: allow `tokens` to go negative (down to a sensible safety bound like `-cap * 4` or unbounded) and call `_refill` before applying `delta` in Lua so elapsed time isn't lost or double-credited.

---

### 2.3 Hardcoded Demo API Keys Active in Production & Missing DB Tenant Lookup
* **Severity:** **Critical**
* **Evidence:**
  * [`gateway/server.py#L28-L40`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/server.py#L28-L40): When `TENANTS_JSON` is unset, `_load_tenants()` registers `"sk-paid-demo"` (`PriorityClass.PAID`, `120_000` TPM) and `"sk-free-demo"` (`PriorityClass.FREE`, `20_000` TPM) with **no `ENVIRONMENT == "production"` guard** (unlike `CACHE_SALT_SECRET` at [`gateway/server.py#L80-L83`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/server.py#L80-L83)).
  * [`gateway/server.py#L94`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/server.py#L94): `RedisPostgresAdapter` is wired with `pg_lookup=keys_by_hash.get` (static boot-time dict, not PostgreSQL).
  * [`deploy/generated/rollout-request-control.yaml#L60-L65`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/rollout-request-control.yaml#L60-L65): The Kubernetes `Rollout` sets **only** `CACHE_SALT_SECRET` and omits `TENANTS_JSON`, `ENVIRONMENT`, `REDIS_URL`, and `GATEWAY_REPLICAS`.
* **Failure Scenario:** Deploying the generated `Rollout` manifest to EKS starts `request-control` with `sk-paid-demo` and `sk-free-demo` enabled globally, `InMemoryBudgetAdapter` active per pod (2x split-brain quota), and no Postgres key rotation/revocation.
* **Remediation:**
  1. Fail closed at startup (`RuntimeError`) if `is_production` and neither `TENANTS_JSON` nor `DATABASE_URL` is configured.
  2. Implement a real SQLAlchemy/Postgres `pg_lookup` querying a `tenants` table (with `key_hash`, `revoked_at`, `tpm_limit`, `max_concurrency`) and add explicit cache invalidation on key revocation rather than relying solely on the 300s Redis TTL ([`gateway/tenant_budget.py#L239`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L239)).

---

### 2.4 `vllm-local` (`v0.7.3`) Silently Ignores `cache_salt` (CVE-2025-46570 Cross-Tenant Timing Leak)
* **Severity:** **High**
* **Evidence:**
  * [`docker-compose.yml#L41`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/docker-compose.yml#L41): Uses `vllm/vllm-openai:v0.7.3`, whereas [`deploy/manifest_renderer.py#L18`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L18) uses `vllm/vllm-openai:v0.9.0`.
  * [`gateway/request_control.py#L608-L611`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L608-L611): Injects `"cache_salt"` into the request JSON.
* **Failure Scenario:** Per [vLLM Security Advisory GHSA-4qjh-9fv9-r85r / CVE-2025-46570](https://github.com/vllm-project/vllm/commit/77073c77bc2006eb80ea6d5128f076f5e6c6f54f) and [vLLM APC Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html), `cache_salt` was added in **`v0.9.0`**. Because vLLM's OpenAI models permit extra fields (`ConfigDict(extra="allow")`), `v0.7.3` accepts `cache_salt` without error and **silently ignores it**, disabling tenant prefix-cache isolation in local Compose testing. Additionally, as [vLLM Security Docs](https://docs.vllm.ai/en/latest/security/index.html) warn, `cache_salt` mitigates KV prefix timing leaks but is **not** a complete GPU/scheduler isolation boundary—regulated/high-security tenants should use `IsolationMode.DEDICATED` pools.
* **Remediation:** Pin `docker-compose.yml` to `vllm/vllm-openai:v0.9.0` (matching Kubernetes manifests) and add an integration test asserting `cache_salt` isolates TTFT hits on `v0.9.0+`.

---

### 2.5 Unvalidated `x-trace-id` Header & Missing `event_id` Fallback Risk
* **Severity:** **High**
* **Evidence:**
  * [`gateway/request_control.py#L670`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L670): `trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex`
  * [`gateway/request_control.py#L358, #L397, #L580`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L580): `trace_id` is echoed in `x-trace-id` response headers and forwarded as `x-request-id` to vLLM.
  * [`gateway/usage_rollup.py#L217`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L217): `dedup_id = decoded.get("event_id") or decoded["trace_id"]` (where [`gateway/request_control.py#L571`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L571) generates `event_id=uuid.uuid4().hex`).
* **Failure Scenario:** While `event_id` prevents caller-controlled `x-trace-id` collisions from dropping billing rows in `RequestControl`, `trace_id` itself is unvalidated: a client can inject arbitrarily long strings, log-forging characters, or duplicate IDs across requests into `x-request-id` (vLLM) and `x-trace-id` logs/headers.
* **Remediation:** Validate `x-trace-id` against a strict regex (`^[A-Za-z0-9_-]{8,64}$`) or generate a gateway-owned `request_id = uuid.uuid4().hex` for internal `x-request-id`/billing while logging caller `x-trace-id` separately after sanitization.

---

### 2.6 Shell Injection & Path Traversal Surfaces in Catalog -> LoRA / Weight Prefetch
* **Severity:** **High**
* **Evidence:**
  * [`catalog/pool_catalog.py#L28-L35`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L28-L35): `LoraAdapterSpec.name` and `s3_uri` (and `ModelPoolSpec.name` / `prefetch_uri`) have **no character/format validation**.
  * [`gateway/request_control.py#L799`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L799) & [`catalog/pool_catalog.py#L160, #L223`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L160): Construct `f"/models/loras/{target_pool.name}/{lora_spec.name}"`.
  * [`deploy/manifest_renderer.py#L367-L388`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L367-L388): Interpolates `rel` and `uri` directly into a `/bin/sh -c` command string inside `daemonset-weight-prefetch.yaml`.
* **Failure Scenario:** A `../` in `lora.name` escapes `/models/loras/{pool}/` (allowing reading weights from another pool directory on disk), and any shell metacharacter (`'`, `"`, `$()`, `;`) in `catalog/pools.yaml` injects arbitrary commands into the `weight-prefetch` DaemonSet pod (which holds S3 IAM credentials and `hostPath` RW access to `/mnt/nvme/models`).
* **Remediation:**
  1. Enforce strict Pydantic validators on `pool.name` and `lora.name` (`^[a-z0-9][a-z0-9._-]{0,62}$`, rejecting `..` and `/`) and validate `s3_uri` / `prefetch_uri` (`^s3://[a-z0-9.-]+/[A-Za-z0-9._/-]+$`).
  2. Pass `sync_entries` via JSON/ConfigMap or environment variables into a parameterized script (never string-interpolated into `/bin/sh -c`).
  3. Mount `model-cache` as **`readOnly: true`** in vLLM Deployments (`_vllm_deployment`) once prefetched, and restrict `VLLM_ALLOW_RUNTIME_LORA_UPDATING` behind `NetworkPolicy` (which [vLLM LoRA Docs](https://docs.vllm.ai/en/latest/models/lora.html) explicitly warn should never be exposed without strict gateway isolation).

---

### 2.7 Unauthenticated `/metrics` Endpoint Exposing Tenant IDs & Missing Request Size Limits
* **Severity:** **Medium-High**
* **Evidence:**
  * [`gateway/request_control.py#L662-L667, #L909`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L909): `GET /metrics` is served unauthenticated on `:8080` alongside `POST /v1/chat/completions`, exposing `gateway_tokens_total{tenant_id=..., pool=...}` and `gateway_429_total{tenant_id=..., reason=...}`.
  * [`gateway/request_control.py#L672`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L672): `await request.json()` has no maximum body size limit (allowing multi-hundred-MB JSON OOM DoS).
* **Remediation:** Serve `/metrics` on a separate internal admin/metrics port (e.g., `:9091`) firewalled by `NetworkPolicy` to only the `monitoring` namespace, and enforce a `Content-Length` / streaming body size cap (e.g., 2–10 MiB) before `request.json()`.

---

## 3. Correctness, Concurrency & Kubernetes / Deploy Gaps

### 3.1 Synchronous Blocking Redis Calls on the Async `uvicorn` Event Loop
* **Severity:** **High**
* **Evidence:**
  * [`gateway/server.py#L91`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/server.py#L91): Instantiates synchronous `redis.Redis.from_url(...)` and passes it to `RedisPostgresAdapter`, `RedisPoolHealthAdapter`, and `RedisStreamUsageSink`.
  * [`gateway/request_control.py#L697, #L873, #L1020, #L1110`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L697): Invokes `budget.authorize_and_reserve`, `budget.reconcile`, `pool_health.get_state`, and `on_usage` (`xadd`) synchronously inside `async def chat_completions` and `async def event_stream`.
* **Failure Scenario:** A single Redis TCP stall or packet drop (default socket timeout unset / OS TCP timeout) freezes the entire `asyncio` event loop thread, halting all active SSE streams and health checks across every tenant on that worker.
* **Remediation:** Migrate gateway runtime Redis clients to `redis.asyncio.Redis` (with explicit `socket_connect_timeout=0.5`, `socket_timeout=1.0`) or wrap synchronous calls via `anyio.to_thread.run_sync`, keeping `BillingRollupWorker` (standalone sync worker) separate.

---

### 3.2 Streaming & Non-Streaming Protocol Gaps (`tool_calls`, `reasoning_content`, Mid-Stream Errors)
* **Severity:** **High**
* **Evidence:**
  * [`gateway/request_control.py#L369, #L893`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L369): `VLLMAdapter.complete` reads `data["choices"][0]["message"]["content"]` (crashes/returns `None` on tool-call responses where `content` is `null`) and hardcodes `"finish_reason": "stop"` (hiding `"length"` or `"tool_calls"`).
  * [`gateway/request_control.py#L430-L438, #L965-L969`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L430-L438): `VLLMAdapter.stream` and `_streaming_chat_response` only extract `delta.get("content")` and `delta.get("role")`. Tool-call deltas (`tool_calls`) and Qwen/DeepSeek `reasoning_content` deltas are dropped, `completion_count` stays `0` on disconnect, and TTFT is mismeasured (`bool(chunk.delta)` at `#L951`).
  * [`gateway/request_control.py#L405, #L945-L1035`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L405): `StreamingResponse` commits HTTP `200 OK` headers before `VLLMAdapter.stream` executes `resp.raise_for_status()`. If vLLM returns `500`/`400` or disconnects mid-stream, `event_stream()` raises unhandled exceptions (abrupt TCP reset, no SSE error frame) and `finally:` still bills `est_prompt + emitted_tokens` (`#L1017-L1029`).
* **Remediation:** Preserve full `delta` fields (`content`, `reasoning_content`, `tool_calls`) and upstream `finish_reason`; preflight/open the upstream stream (reading status code before returning `StreamingResponse`) so upstream `4xx`/`5xx` returns a clean HTTP error with zero billing; catch stream exceptions inside `event_stream()` to emit an OpenAI-style SSE error chunk and reconcile `0` tokens if zero tokens were generated on an upstream failure.

---

### 3.3 `TenantBudget` Concurrency Leak, Stale `_tenant_caps`, & Reserve-Before-Authz Ordering
* **Severity:** **High**
* **Evidence:**
  * [`gateway/tenant_budget.py#L240`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L240): `permit_ttl_seconds = 120.0`, equal to `VLLMAdapter` read timeout (`120.0s`). Requests streaming >120s have their concurrency permit pruned early by `ZREMRANGEBYSCORE` (`#L176`), letting tenants exceed `max_concurrency`.
  * [`gateway/tenant_budget.py#L256-L275, #L341`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L256-L275): `lookup_tenant()` on a Redis cache hit (`#L260-L267`) does not populate `self._tenant_caps`, and `reconcile()` reads `self._tenant_caps.get(permit.tenant_id, max(permit.reserved_tokens, 1))`—using stale process-local caps or clamping to `reserved_tokens`.
  * [`gateway/tenant_budget.py#L352-L353`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py#L352-L353): When `reserve()` succeeded in Redis (`degraded_mode=False`) and `reconcile()` hits `RedisError`, `reconcile()` calls `self._local_fallback.reconcile(...)` (a no-op!), leaving a "phantom" concurrency slot stuck in Redis `conc:{tenant_id}` for up to 120s (mirroring the [LiteLLM phantom budget exhaustion bug](https://github.com/BerriAI/litellm)).
  * [`gateway/request_control.py#L697-L739`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L697-L739): Reserves TPM/concurrency *before* checking `lora_spec.tenant_id`, and records `tenant_id="unknown"` on `BudgetExceededError` (`#L713`).
* **Remediation:** Increase `permit_ttl_seconds` above max request duration (e.g. `600s`) with stream heartbeats or strict request timeout; pass `cap` in `Permit` instead of `_tenant_caps`; authenticate and verify LoRA ownership before `reserve()`; attach `tenant_id`/`priority` to `BudgetExceededError`; and queue failed Redis reconciliations for retry.

---

### 3.4 `BillingRollupWorker` Poison-Pill Starvation, N+1 DB Inserts, & Shutdown Data Loss
* **Severity:** **High**
* **Evidence:**
  * [`gateway/usage_rollup.py#L176-L190`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L176-L190): `recover_stale()` hardcodes `start_id="0-0"` and ignores `result[0]` (next cursor) and delivery counts (`XPENDING`).
  * [`gateway/usage_rollup.py#L242-L254`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L242-L254): Any single Postgres error (e.g., `\x00` byte in text, constraint violation) rolls back the entire batch and leaves it un-`XACK`ed, creating an infinite poison-pill loop that starves entries beyond index 100.
  * [`gateway/usage_rollup.py#L249-L252`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L249-L252): Executes 1 SQL `INSERT` round-trip per event in a Python loop (N+1) and has no retention/partitioning on `usage_event_dedup`.
  * [`gateway/usage_rollup.py#L88`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L88) & [`gateway/request_control.py#L658-L660`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L658-L660): `deque(maxlen=4096)` silently drops oldest events on overflow; `on_usage_flush()` suppresses all exceptions during shutdown.
  * [`pyproject.toml#L6-L17`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/pyproject.toml#L6-L17) & [`gateway/usage_rollup.py#L325`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/usage_rollup.py#L325): Missing `psycopg[binary]` dependency and no `postgresql://` -> `postgresql+psycopg://` dialect normalization for SQLAlchemy 2.0.
* **OSS Benchmark Reference:** [Redis Streams Docs](https://redis.io/docs/latest/develop/data-types/streams/) prescribe inspecting delivery counts (`XPENDING` / `XAUTOCLAIM` cursor iteration) and routing entries exceeding max deliveries to a dead-letter stream (`XADD ...-dlq`) + `XACK`.
* **Remediation:** Advance `XAUTOCLAIM` using `result[0]`, check `XPENDING` delivery count to route poison pills (`>= 5` deliveries) to a DLQ stream, batch `usage_event_dedup` inserts into a single statement, add `psycopg[binary]` to `pyproject.toml`, and log/metric dropped buffer events.

---

### 3.5 `SelfHealingEngine` & `PoolHealth` Logic Bugs and Missing Wiring
* **Severity:** **Medium-High**
* **Evidence:**
  * [`controller/self_healing.py#L255-L257, #L318-L338`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L318-L338): `reconcile()` adds nodes to `guardrails.draining_nodes` on `CORDON_AND_DRAIN`, but never calls `complete_drain()`, permanently blocking all future drains across the cluster (`max_one_drain`).
  * [`controller/self_healing.py#L290-L292`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L290-L292): `pod.kv_cache_usage >= 0.95` triggers `RESTART_POD` (`delete_pod`), wiping active KV caches and aborting in-flight streams under high legitimate traffic.
  * [`controller/self_healing.py#L220-L221`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L220-L221): When `mad < 1e-6`, `compute_z_score` uses `abs(series[-1] - med)` (`latest_diff >= min_delta`) and returns `+10.0` even on a latency *decrease* (false anomaly on speedup).
  * [`controller/self_healing.py#L355-L368`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L355-L368): After `max_failures_before_page`, `page_human` fires on every subsequent tick without an escalation cooldown.
  * [`gateway/pool_health.py#L158, #L363`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/pool_health.py#L158): `record_probe` is never wired into `request_control.py` or `server.py`, and `SelfHealingEngine` has no runner/K8s RBAC/Deployment (`kubernetes` client is absent from `pyproject.toml`).

---

### 3.6 Kubernetes, Argo Rollouts, KEDA, Karpenter & Terraform Deployment Blockers
* **Severity:** **High**
* **Evidence:**
  * **KEDA Scale-from-Zero Deadlock** ([`deploy/generated/keda-pool-fallback.yaml#L11-L26`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/keda-pool-fallback.yaml#L11-L26)): `minReplicaCount: 0` triggers on `vllm:num_requests_waiting`, which does not exist in Prometheus when pods are at `0` ([KEDA Prometheus Scaler Docs](https://keda.sh/docs/latest/scalers/prometheus/)). Must trigger on gateway-side demand (e.g., `gateway_fallback_total` or gateway queue metric).
  * **Argo Rollouts Canary Misconfiguration** ([`deploy/generated/rollout-request-control.yaml#L33-L87`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/rollout-request-control.yaml#L33-L87)): `replicas: 2` with `setWeight: 5` and `25` has **no `trafficRouting`** (so Argo approximates weight by pod count—0% or 50%, never 5% per [Argo Rollouts Docs](https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/)), while `AnalysisTemplate` evaluates global `gateway_ttft_seconds_bucket` across stable+canary combined against `pool-fallback`'s `<= 0.600` SLO (causing false rollbacks on `pool-8b`'s 1.0s SLO).
  * **NetworkPolicy Egress & Missing Policies** ([`deploy/generated/networkpolicy-inference.yaml#L28-L36`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/networkpolicy-inference.yaml#L28-L36)): Blocks Hugging Face (`443`), S3 public IPs, and EKS `172.20.0.0/16` Service CIDR; leaves `request-control`, `weight-prefetch`, Redis, and Postgres unprotected.
  * **Pod Security Standards Violations** ([`deploy/generated/deployment-pool-8b.yaml#L29-L97`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/deployment-pool-8b.yaml#L29-L97), [`daemonset-weight-prefetch.yaml#L35-L80`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/daemonset-weight-prefetch.yaml#L35-L80), [`rollout-request-control.yaml#L66-L70`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/rollout-request-control.yaml#L66-L70)): Missing `capabilities.drop: ["ALL"]` and `seccompProfile: RuntimeDefault` required by [Kubernetes PSS Restricted](https://kubernetes.io/docs/concepts/security/pod-security-standards/); `hostPath` volumes forbidden under Baseline/Restricted; `dcgm-daemonset.yaml` lacks `CAP_SYS_ADMIN`.
  * **Missing K8s Services & Instance Sizing Mismatch**: No `Service` objects exist for `vllm-pool-*` or `dcgm-exporter` (breaking `prometheus-scrape.yaml` and `VLLM_ENDPOINTS`); `16Gi`/`4` vCPU requests in `deployment-pool-8b.yaml#L53-L54` cannot fit on `g6.xlarge`/`g5.xlarge` (`16 GiB` total RAM) allowed by [`karpenter-gpu.yaml#L24-L25`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/generated/karpenter-gpu.yaml#L24-L25).
  * **Terraform & S3 Bucket Skew** ([`deploy/terraform/main.tf#L47, #L63-L72`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/terraform/main.tf#L47), [`deploy/terraform/variables.tf#L13`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/terraform/variables.tf#L13), [`deploy/manifest_renderer.py#L20`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L20)): Bucket name mismatch (`multi-tenant-llm-models-prod` vs `llm-model-weights`); public EKS endpoint; missing S3 public-access block/KMS, Karpenter IAM/SQS, IRSA/Pod Identity, RDS, and ElastiCache.

---

### 3.7 `BenchmarkRunner` Regression Gate Blind Spot
* **Severity:** **Medium**
* **Evidence:** [`bench/runner.py#L315-L335`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/bench/runner.py#L315-L335): `_evaluate_verdict` only fails on `report.succeeded == 0`, `ttft_ms_p95 > slo_p95_ttft_ms`, or throughput regression.
* **Failure Scenario:** If 49 of 50 requests fail (`HTTP 500`/`503`) and 1 succeeds under SLO, `verdict.passed` is `True`! It also omits `availability` / error-rate gating, `itl_ms_p95`, and multi-tenant noisy-neighbor concurrency profiles, and [`bench/cli.py#L22-L31`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/bench/cli.py#L22-L31) omits `--warmup-requests`, `--system-prompt`, and `--slo-p95-ttft-ms`.

---

## 4. High-Value OSS Patterns to Adopt

1. **Prefix-Cache Scoring & Full-Prompt Block Hashing ([`vllm-project/aibrix`](https://github.com/vllm-project/aibrix), [`kubernetes-sigs/gateway-api-inference-extension`](https://github.com/kubernetes-sigs/gateway-api-inference-extension), [`llm-d/llm-d`](https://github.com/llm-d/llm-d)):**
   * Replace the 256-char `_extract_prefix` heuristic (`request_control.py#L1039-L1043`) with multi-block prefix hashing over the canonicalized message/tool prefix (aligned to vLLM's 16-token block size) and soft-blended scoring (`prefix_hit_score` + `requests_waiting` + `kv_cache_usage`), or adopt GAIE `InferencePool` + Envoy `ext-proc` EPP.
2. **LoRA-Resident Routing & Active Adapter Tracking ([`vllm-project/aibrix`](https://aibrix.readthedocs.io/), [`gateway-api-inference-extension`](https://github.com/kubernetes-sigs/gateway-api-inference-extension)):**
   * Replace per-gateway-pod `_loaded_loras` sets with shared Redis/endpoint LoRA residency state (polling `/v1/models` or sidecar telemetry) and `/v1/unload_lora_adapter` lifecycle management, routing dynamic LoRA requests preferentially to replicas where the adapter is already loaded.
3. **Gateway-Level Token Cost Expressions & Redis Async Pipelining ([`envoyproxy/ai-gateway`](https://github.com/envoyproxy/ai-gateway), [`BerriAI/litellm`](https://github.com/BerriAI/litellm)):**
   * Track `cached_prompt_tokens` (`prompt_tokens_details.cached_tokens`), `reasoning_tokens`, `input_tokens`, and `output_tokens` separately for differential billing and rate limiting, using async Redis Lua scripts.
4. **LMCache / Disaggregated Prefill-Decode Path ([`vllm-project/production-stack`](https://github.com/vllm-project/production-stack), [`llm-d/llm-d`](https://github.com/llm-d/llm-d)):**
   * Add optional CPU/NVMe KV offloading (`--kv-offloading-backend lmcache`) for long-context shared system prompts when GPU HBM is constrained on 24 GB L4/A10G nodes.

---

## 5. Repo & CI/CD Hygiene Checklist

| Check | Current State | Recommended Fix |
| :--- | :--- | :--- |
| **GitHub Actions CI (`.github/workflows/`)** | **Missing** (`Test-Path .github == False`). | Add CI workflow running `pytest`, `ruff check .`, `mypy`, `pip-audit`, `trivy` container/IaC scan, and `deploy/render_all.py` drift check (`git diff --exit-code deploy/generated`). |
| **Container Supply Chain (`Dockerfile`, `deploy/manifest_renderer.py`)** | Unpinned `python:3.12-slim`, `request-control:latest`, `busybox:1.36`, `alpine:3.20`; missing `psycopg[binary]`. | Pin base/sidecar images by immutable `@sha256:...` digest, tag `request-control` by commit SHA/semver, add `psycopg[binary]` to `pyproject.toml`, generate SBOM + provenance. |
| **Python Version Alignment (`pyproject.toml`)** | `requires-python = ">=3.12"` (`#L5`), `Dockerfile` uses `3.12`, `[tool.mypy] python_version = "3.14"` (`#L27`), unused `sklearn`/`yaml.*` overrides (`#L33`). | Align `mypy` `python_version = "3.12"` (matching container runtime) and clean up unused `mypy` override entries. |
| **`.gitignore` Secret & IaC Protection** | Only ignores Python caches & build dirs (`.gitignore#L1-L9`). | Add `.env`, `.env.*`, `*.tfstate`, `*.tfstate.*`, `.terraform/`, `*.tfvars`, `*.pem`, `*.key`. |
| **Working Tree Hygiene** | Untracked `docs/research/live-gpu-postgres-eks-readiness.md` (`git status`); outdated `docs/research/present-architecture-primary-review.md` still present. | Commit/archive or link research docs consistently and mark superseded audits clearly. |

---

## 6. Sources

1. **vLLM Automatic Prefix Caching & `cache_salt`:** <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
2. **vLLM Security Documentation:** <https://docs.vllm.ai/en/latest/security/index.html>
3. **vLLM CVE-2025-46570 / GHSA-4qjh-9fv9-r85r Patch (`v0.9.0`):** <https://github.com/vllm-project/vllm/commit/77073c77bc2006eb80ea6d5128f076f5e6c6f54f>
4. **vLLM LoRA Adapters & `VLLM_ALLOW_RUNTIME_LORA_UPDATING`:** <https://docs.vllm.ai/en/latest/models/lora.html>
5. **vLLM OpenAI-Compatible Server Protocol:** <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
6. **AIBrix (`vllm-project/aibrix`) GitHub & Docs:** <https://github.com/vllm-project/aibrix> and <https://aibrix.readthedocs.io/>
7. **vLLM Production Stack (`vllm-project/production-stack`):** <https://github.com/vllm-project/production-stack>
8. **Gateway API Inference Extension (`kubernetes-sigs/gateway-api-inference-extension`):** <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
9. **llm-d (`llm-d/llm-d` & `llm-d/llm-d-router`):** <https://github.com/llm-d/llm-d> and <https://github.com/llm-d/llm-d-router>
10. **Envoy AI Gateway (`envoyproxy/ai-gateway`):** <https://github.com/envoyproxy/ai-gateway> and <https://aigateway.envoyproxy.io/>
11. **LiteLLM (`BerriAI/litellm`) Proxy & Rate Limiting:** <https://github.com/BerriAI/litellm>
12. **Redis Streams (`XAUTOCLAIM`, `XPENDING`, Poison Pills / DLQ):** <https://redis.io/docs/latest/develop/data-types/streams/>
13. **KEDA Prometheus Scaler (`activationThreshold`, scale-to-zero):** <https://keda.sh/docs/latest/scalers/prometheus/>
14. **Argo Rollouts Canary & `trafficRouting`:** <https://argo-rollouts.readthedocs.io/en/stable/features/traffic-management/>
15. **Kubernetes Pod Security Standards (Restricted Profile):** <https://kubernetes.io/docs/concepts/security/pod-security-standards/>
