# Primary-Source Audit of the Present Executed Architecture

Scope: `catalog/pool_catalog.py`, `gateway/pool_health.py`, `gateway/tenant_budget.py`, `gateway/request_control.py`, and `controller/self_healing.py`.

Each finding below is mapped to exact file paths and line numbers and classified as:
- **Correct as built**: Matches production specs and invariants as currently implemented.
- **Production-gap / bug under multi-replica**: Incorrect behavior, spec mismatch, resource leak, or multi-pod invariant break in production.
- **Algorithmic upgrade**: Works functionally as a baseline, but primary literature and production serving stacks offer materially stronger algorithms.

---

## Executive Summary Matrix

| Area | Classification | Code Location | Summary |
| :--- | :--- | :--- | :--- |
| Catalog validation & DAG check | **Correct as built** | `catalog/pool_catalog.py#L41-L47`, `#L133-L148` | Validates replica bounds, non-scale-to-zero shared pools, unique aliases, and acyclic fallback chains. |
| Pool health hysteresis & protocol seam | **Correct as built** | `gateway/pool_health.py#L28-L35`, `#L89-L121` | Clean protocol seam; anti-flap cooldown and consecutive-probe recovery; `UNHEALTHY` not overwritten by `set_degraded`. |
| Pool health cross-pod sync & TTL expiry | **Production-gap / bug under multi-replica** | `gateway/pool_health.py#L48-L87`, `#L110` | Process-local `_records` invisible across gateway/controller pods; `get_state` never auto-expires `override_until` without probes. |
| SHA-256 key hashing & reserve/reconcile | **Correct as built** | `gateway/tenant_budget.py#L56-L58`, `#L121-L127`, `#L180` | Never stores plaintext API keys; clamps reconciled bucket tokens to `[0, cap]`. |
| Redis Lua `HMSET`/`HMGET` & missing reject TTL | **Production-gap / bug under multi-replica** | `gateway/tenant_budget.py#L149-L169` | `HMSET` deprecated since Redis 4.0 in favor of `HSET`; rejected requests write without `EXPIRE`. |
| Redis Lua caller clock vs server `TIME` | **Production-gap / bug under multi-replica** | `gateway/tenant_budget.py#L146`, `#L196`, `#L242-L249` | Caller `time.time()` exposes bucket refill to inter-pod clock skew; Redis 5+/7+ effects replication allows `redis.call('TIME')`. |
| Per-pod `_active_permits` concurrency cap | **Production-gap / bug under multi-replica** | `gateway/tenant_budget.py#L301`, `#L312-L326` | Process-local set multiplies effective cap to `N * max_concurrency` behind an N-pod K8s Service; lacks Redis atomic lease/TTL. |
| Degraded mode `degraded_tpm_ratio` scaling | **Production-gap / bug under multi-replica** | `gateway/tenant_budget.py#L194`, `#L268-L274` | Unscaled `0.25` per pod yields `N * 0.25 * tpm_limit` cluster-wide during Redis outage (`>100%` when `N > 4`). |
| `Retry-After` header semantics on `429` | **Correct as built** | `gateway/request_control.py#L403`, `#L413` | Positive integer `delay-seconds` conforms to RFC 9110 §10.2.3 and OpenAI rate-limit guidance. |
| OpenAI schema, error envelope & vLLM stream/abort | **Production-gap / bug under multi-replica** | `gateway/request_control.py#L268-L294`, `#L387-L424`, `#L456-L513` | `VLLMAdapter.stream` double-wraps raw SSE JSON; `abort` calls non-existent `/v1/abort`; chunks miss `created`/`finish_reason`/`usage`; flat `{"error": "..."}` envelope. |
| `_PriorityAdmission` cancellation & budget leak | **Production-gap / bug under multi-replica** | `gateway/request_control.py#L125-L157`, `#L406-L415`, `#L515-L523` | `CancelledError` while queued skips `budget.reconcile(permit, 0)`; post-wakeup cancellation leaks `_in_flight`; `finally` abort lacks AnyIO cancel-shielding. |
| Prompt token estimator (`str.split()`) | **Algorithmic upgrade** | `gateway/request_control.py#L297-L302` | Word splitting underestimates Qwen2.5 BPE tokens on code, JSON, ChatML overhead, and non-Latin/CJK text. |
| Modulo prefix hashing & fallback load pick | **Algorithmic upgrade** | `gateway/request_control.py#L541-L560` | `hash % len(ordered)` reshuffles ~`(N-1)/N` keys on scale events; replace with Consistent Hashing with Bounded Loads (`1 + epsilon`). |
| Guardrail & actuator seam architecture | **Correct as built** | `controller/self_healing.py#L44-L92`, `#L215-L266` | Enforces `dry_run`, per-target cooldown, `max_one_drain`, and failure-count human escalation via `ClusterActuator`. |
| K8s Cordon/Drain `policy/v1` Eviction & multi-pod controller | **Production-gap / bug under multi-replica** | `controller/self_healing.py#L118-L140`, `#L175-L177` | `K8sActuatorAdapter` lacks `node.spec.unschedulable` + `policy/v1` Eviction (`429` PDB retry loop); `GuardrailState` is in-memory without leader election. |
| Sample mean/std Z-score & univariate telemetry | **Algorithmic upgrade** | `controller/self_healing.py#L21-L33`, `#L142-L152` | Sample mean/std masks repeat spikes and false-positives on `std < 1e-6`; upgrade to MAD Modified Z-Score and `IsolationForest`. |

---

## 1. Model Pool Catalog (`catalog/pool_catalog.py`)

- **Correct as built — Invariant & fallback graph validation (`catalog/pool_catalog.py#L41-L47`, `#L106-L148`)**:
  - `ModelPoolSpec._validate_pool_invariants` enforces `min_replicas <= max_replicas` and prevents scale-to-zero (`min_replicas=0`) for `IsolationMode.SHARED` pools (`#L43-L46`).
  - `load_and_validate` guarantees unique alias mapping (`#L122-L127`), and `_validate_fallback_graph` verifies every `fallback_pool` target exists and contains no cycles (`#L133-L148`).
- **Production-gap / bug under multi-replica — Static startup snapshot (`catalog/pool_catalog.py#L50-L103`)**:
  - `ModelPoolCatalog` is instantiated as an in-memory object passed into `create_app` (`gateway/request_control.py#L362`). Multi-replica gateway pods have no built-in reload or ConfigMap watch mechanism when pool definitions or `fallback_pool` targets change at runtime, and `render_targets` (`#L66-L103`) emits generic dicts rather than Kubernetes KEDA `ScaledObject` manifests.

---

## 2. Pool Health Store (`gateway/pool_health.py`)

- **Correct as built — Hysteresis & protocol seam (`gateway/pool_health.py#L28-L35`, `#L77-L121`)**:
  - `PoolHealth` defines a clean Protocol boundary (`get_state`, `set_degraded`, `record_probe`) between fast-path `RequestControl` and slow-path `SelfHealingEngine`.
  - `PoolHealthStore` prevents flapping via `recovery_cooldown_seconds` (`#L110`), `min_healthy_probes` (`#L115`), `unhealthy_failure_threshold` (`#L96`), and preserves `HealthState.UNHEALTHY` when `set_degraded` is called (`#L81-L83`).
- **Production-gap / bug under multi-replica — Process-local `_records` (`gateway/pool_health.py#L48-L63`, `controller/self_healing.py#L275-L276`)**:
  - `PoolHealthStore._records` is a Python `dict` in local process memory (`#L63`), using `time.monotonic` (`#L57`). In Kubernetes, `SelfHealingEngine` runs in a controller pod while `N` gateway pods run behind a `Service`; calling `self._pool_health.set_degraded(...)` (`controller/self_healing.py#L276`) updates only the controller's local dict and never reaches gateway routing (`gateway/request_control.py#L574`).
  - **Fix**: Implement a shared `PoolHealth` adapter backed by Redis (`HSET` + `EXPIRE` using server `TIME` or epoch timestamps) or Kubernetes custom resources / ConfigMap watches.
- **Production-gap / bug under multi-replica — Silent `override_until` lock-in (`gateway/pool_health.py#L65-L87`, `#L110`)**:
  - `set_degraded` sets `rec.override_until = now + ttl_seconds` (`#L84`), which blocks recovery inside `record_probe(..., ok=True)` (`#L110`). However, `get_state` (`#L65-L75`) never checks whether `now >= rec.override_until`. If a pool is marked degraded via `set_degraded` and no active prober calls `record_probe(pool, ok=True)` after the TTL expires, the pool remains `DEGRADED` indefinitely.

---

## 3. Redis Lua Token Bucket, Concurrency & Degraded Mode (`gateway/tenant_budget.py`)

- **Correct as built — Key hashing & reserve/reconcile accounting (`gateway/tenant_budget.py#L56-L58`, `#L102-L128`, `#L180`, `#L206-L237`)**:
  - `hash_api_key` hashes credentials with SHA-256 (`#L56-L58`) before caching in Redis (`tenant_key:{key_hash}`) or looking up in Postgres.
  - Pre-request reservation of `est_prompt + max_tokens` (`gateway/request_control.py#L390`) followed by post-completion reconciliation clamping tokens to `[0, cap]` (`#L127`, `#L180`) prevents quota bypass by streaming or long-running requests.

- **Production-gap / bug under multi-replica — Deprecated `HMSET` & missing TTL on rejected writes (`gateway/tenant_budget.py#L149-L169`)**:
  - `_RESERVE_LUA` calls `HMGET` (`#L149`) and `HMSET` (`#L162`, `#L167`). Per the official Redis documentation, `HMSET` has been **deprecated since Redis 4.0.0** ([Redis `HMSET`](https://redis.io/docs/latest/commands/hmset/)) and superseded by multi-field `HSET` ([Redis `HSET`](https://redis.io/docs/latest/commands/hset/)), which `_RECONCILE_LUA` already uses (`#L181`).
  - In the rejection branch (`if tokens < req then`, `#L161-L164`), `_RESERVE_LUA` persists the refilled `tokens` and `ts` (`#L162`) without calling `EXPIRE` (unlike `#L168` on the success path). If a brand-new or expired tenant bucket sends a request exceeding `cap`, Redis stores the hash key with no TTL, leaking memory across churning keys.

- **Production-gap / bug under multi-replica — Caller wall-clock `time.time()` vs Redis `TIME` (`gateway/tenant_budget.py#L146`, `#L156`, `#L196`, `#L242-L249`)**:
  - `RedisPostgresAdapter` passes the gateway pod's local `time.time()` as `ARGV[2]` (`#L196`, `#L242`, `#L249`). Across `N` gateway pods, NTP drift or clock skew causes Pod A (ahead by $\Delta t$) to write a future `ts`, forcing Pod B's `math.max(0.0, now - ts)` (`#L156`) to evaluate to `0.0` (stalling bucket refill) or letting skewed pods over-refill.
  - Historically, older Redis verbatim script replication restricted calling non-deterministic commands like `TIME` ([Redis `TIME`](https://redis.io/docs/latest/commands/time/)) before writes in Lua scripts. However, **script effects replication** became the default in Redis 5.0 and **exclusive in Redis 7.0+** (removing verbatim script replication), so Redis replicates only the deterministic write commands produced by the script ([Redis `EVAL` Intro](https://redis.io/docs/latest/develop/programmability/eval-intro/)). Calling `local t = redis.call('TIME')` inside Lua (`t[1] + t[2] / 1000000.0`) provides a single monotonic-enough cluster clock across all gateway replicas.

- **Production-gap / bug under multi-replica — Process-local concurrency cap (`gateway/tenant_budget.py#L301`, `#L312-L326`)**:
  - `TenantBudget._active_permits` (`#L301`) is an in-process `dict[str, set[str]]`. Behind a Kubernetes `Service` with `N` gateway replicas, each pod independently enforces `len(active) >= tenant.max_concurrency` (`#L313`), allowing up to **`N * tenant.max_concurrency`** concurrent requests globally.
  - **Fix**: Enforce concurrency atomically in Redis inside the same Lua call as TPM reservation—either via `INCR`/`DECR` with a safety `EXPIRE` lease ([Redis `EVAL` Intro](https://redis.io/docs/latest/develop/programmability/eval-intro/)) or a Redis `ZSET` keyed by `conc:{tenant_id}` storing `(permit_id, expires_at)` where expired entries from crashed pods (`OOMKilled`/evicted without calling `reconcile`) are pruned via `ZREMRANGEBYSCORE` before checking `ZCARD`.

- **Production-gap / bug under multi-replica — Unscaled Redis outage `degraded_tpm_ratio` (`gateway/tenant_budget.py#L194`, `#L203`, `#L268-L274`)**:
  - When Redis raises `redis.RedisError`, `RedisPostgresAdapter.reserve` delegates to `self._local_fallback` with `tpm_limit = max(1, int(tenant.tpm_limit * self._degraded_ratio))` (`degraded_tpm_ratio = 0.25`, `#L194`, `#L271`). Across `N` gateway replicas, each pod grants 25% locally, summing to **`N * 0.25 * tenant.tpm_limit`** globally (exceeding 100% of the tenant's quota whenever `N > 4`). Both degraded TPM and degraded concurrency limits must be divided by the active gateway replica count `N` (e.g., `degraded_tpm_ratio / replica_count`).

---

## 4. OpenAI Chat Completions, Token Counting & Starlette ASGI Lifecycle (`gateway/request_control.py`)

### 4.1 OpenAI API & vLLM Wire Compatibility

- **Correct as built — `Retry-After` header semantics (`gateway/request_control.py#L403`, `#L413`)**:
  - Both `BudgetExceededError` and `_AdmissionShedError` emit HTTP `429` with `Retry-After` formatted as a positive integer string of seconds (`delay-seconds`), matching RFC 9110 §10.2.3 ([RFC 9110](https://www.rfc-editor.org/rfc/rfc9110)) and OpenAI's rate-limit retry guidance ([OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)). (Adding an optional `Retry-After` on `503` at `#L424` is also permitted by RFC 9110.)

- **Production-gap / bug under multi-replica — `VLLMAdapter.stream` double-wraps SSE JSON & lacks `include_usage` (`gateway/request_control.py#L268-L289`, `#L504-L510`, `#L524`)**:
  - `VLLMAdapter.stream` reads raw SSE lines from vLLM's `/v1/chat/completions` endpoint and yields `StreamChunk(delta=line[6:], completion_tokens_so_far=count)` (`#L286-L288`). Because vLLM emits full OpenAI-compatible JSON objects after `data: `, `line[6:]` is a JSON string, which `_streaming_chat_response` then re-wraps inside `{"choices": [{"index": 0, "delta": {"content": chunk.delta}}]}` (`#L508`), corrupting the streamed output.
  - Additionally, `VLLMAdapter.stream` does not set `stream_options: {"include_usage": true}` (`#L276`), which vLLM and OpenAI use to emit a final accounting chunk with `choices: []` and exact `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) before `data: [DONE]` ([OpenAI Chat API Reference](https://platform.openai.com/docs/api-reference/chat), [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)). Currently, `#L524` reconciles with `est_prompt + emitted_tokens`, where `emitted_tokens` merely counts SSE lines (`#L287`).

- **Production-gap / bug under multi-replica — `VLLMAdapter.abort` calls non-existent `/v1/abort` (`gateway/request_control.py#L290-L294`)**:
  - `VLLMAdapter.abort` issues `POST {replica.url}/v1/abort` (`#L292`). According to vLLM's OpenAI-compatible server documentation ([vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)), stock vLLM exposes **no `/v1/abort` HTTP route**; instead, it detects client HTTP disconnection (`raw_request.is_disconnected()`) on the active `/v1/chat/completions` stream and calls `AsyncLLMEngine.abort(request_id)` internally. Closing the upstream `httpx` stream context (`#L277`) cancels generation on vLLM, whereas `/v1/abort` returns `404` unless backed by a custom sidecar.

- **Production-gap / bug under multi-replica — OpenAI chunk/completion fields & error JSON envelope (`gateway/request_control.py#L382`, `#L387`, `#L398`, `#L401`, `#L411`, `#L424`, `#L456-L475`, `#L504-L513`)**:
  - **Streaming chunks (`#L504-L513`)**: Omit `"created"` (Unix epoch seconds) and `"finish_reason"` (`null` on intermediate chunks, `"stop"` on the terminal chunk before `data: [DONE]\n\n`), and omit an initial role delta (`{"role": "assistant", "content": ""}`) ([OpenAI Chat API Reference](https://platform.openai.com/docs/api-reference/chat)).
  - **Non-streaming completion (`#L456-L475`)**: Omits top-level `"created"`.
  - **Error envelope (`#L387`, `#L398`, `#L401`, `#L411`, `#L424`)**: Returns flat `{"error": str(exc)}` instead of OpenAI's standard nested object `{"error": {"message": ..., "type": ..., "param": null, "code": ...}}` ([OpenAI Error Codes](https://platform.openai.com/docs/guides/error-codes)). Also `ChatCompletionRequest.model_validate(raw_body)` (`#L382`) lacks `ValidationError` / `JSONDecodeError` handling, producing a `500` instead of `400` (`invalid_request_error`) on malformed requests.

### 4.2 Prompt Token Counting

- **Algorithmic upgrade — `_estimate_prompt_tokens` whitespace splitting (`gateway/request_control.py#L297-L302`)**:
  - `_estimate_prompt_tokens` counts `len(content.split())` (`#L301`). Qwen2.5 uses a byte-level BPE tokenizer with a 151,665-token vocabulary and ChatML control tokens (`<|im_start|>`, `<|im_end|>`) ([Qwen Quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)).
  - English prose typically yields ~1.3–1.5 tokens per word (~3.5–4.5 chars/token), code/JSON yields ~2.5–3 chars/token, and CJK/non-Latin scripts without spaces collapse an entire paragraph into `len(content.split()) == 1`, severely underestimating pre-admission TPM usage.
  - **Upgrade**: Pre-tokenize with `transformers.AutoTokenizer` + `apply_chat_template` ([Qwen Quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)) or vLLM's `/tokenize` endpoint, or use a conservative UTF-8 byte-ratio bound (`ceil(len(utf8_bytes) / 3.0) + chatml_overhead`) when avoiding gateway-side tokenizer CPU overhead.

### 4.3 Starlette `StreamingResponse`, Uvicorn `http.disconnect`, and `_PriorityAdmission`

- **How ASGI cancellation works**: Under the ASGI HTTP spec, Uvicorn delivers `{"type": "http.disconnect"}` on `receive()` when the client closes the TCP connection ([ASGI HTTP Spec](https://asgi.readthedocs.io/en/latest/specs/www.html)). Starlette's `StreamingResponse` runs `stream_response` and `listen_for_disconnect` concurrently inside an AnyIO task group; upon receiving `http.disconnect`, Starlette cancels the task group, injecting `asyncio.CancelledError` into `event_stream()` at its next `await` point ([Starlette Responses](https://www.starlette.io/responses/)).

- **Production-gap / bug under multi-replica — Unshielded `abort()` in `finally` (`gateway/request_control.py#L515-L523`)**:
  - While `#L501-L502` polls `await request.is_disconnected()` between chunks and `#L514-L532` cleans up inside `finally:`, lines `#L517-L522` only catch `except RuntimeError:` around `await deps.engine.abort(...)`. In Python 3.8+, `asyncio.CancelledError` inherits from `BaseException`, not `RuntimeError`. When Starlette's AnyIO cancel scope cancels `event_stream()` mid-`await`, any async I/O inside `await deps.engine.abort(...)` is immediately re-cancelled before the HTTP call or fallback `create_task` executes. Cleanup `await` calls inside `finally:` must be wrapped in `anyio.CancelScope(shield=True)` (or `asyncio.shield`).

- **Production-gap / bug under multi-replica — `_PriorityAdmission` cancellation & `TenantBudget` leak (`gateway/request_control.py#L125-L157`, `#L406-L415`)**:
  - **Waiter cancellation cleanup (`#L147-L157`)**: `_PriorityAdmission.acquire` catches `BaseException` and calls `fut.cancel()` (`#L148`). In `release()` (`#L151-L157`), popping until `not fut.done()` avoids `asyncio.InvalidStateError` ([Python `asyncio.Future`](https://docs.python.org/3/library/asyncio-future.html)), because cancelled futures report `done() == True`.
  - **Slot leak #1 (queued client disconnect skips `budget.reconcile`, `#L392-L415`)**: `authorize_and_reserve` runs at `#L392`, *before* `await deps.admission.acquire(permit.priority)` at `#L407`. Lines `#L408-L414` only catch `_AdmissionShedError`. If a client disconnects while waiting in `_PriorityAdmission`, Starlette/Uvicorn cancels the request task (`CancelledError`), bypassing `#L409` and **permanently leaking the tenant's `_active_permits` slot and reserved TPM tokens**.
  - **Slot leak #2 (wakeup-vs-cancel race in `acquire`, `#L145-L156`)**: If `release()` calls `fut.set_result(None)` (`#L155`, transferring the `_in_flight` slot to the waiter without decrementing `_in_flight`), and the waiter's task is cancelled on the same event-loop tick before resuming from `await fut`, `fut.cancel()` returns `False` (the future already has a result), `acquire()` raises `CancelledError`, and `chat_completions` never calls `deps.admission.release()`, **permanently leaking `_in_flight`**.
  - **Fast-path stall on cancelled waiters (`#L126`)**: `if self._in_flight < self._policy.max_in_flight and not self._waiters:` treats a heap containing only cancelled futures as non-empty, forcing subsequent callers into the slow path (and risking an indefinite wait if `_in_flight == 0` when all prior tasks exited).

---

## 5. Prefix-Cache Routing & Bounded-Load Consistent Hashing (`gateway/request_control.py`)

- **Algorithmic upgrade — Modulo hashing (`hash % N`) vs Consistent Hashing with Bounded Loads (`gateway/request_control.py#L541-L560`)**:
  - `_pick_replica` sorts replicas by `replica_id` and selects `idx = int.from_bytes(digest[:8], "big") % len(ordered)` (`#L551-L554`), spilling over to global `min(ordered, ...)` only when `requests_waiting > keda_waiting_threshold` or `kv_cache_usage > keda_kv_cache_threshold` (`#L555-L560`).
  - **Flaw 1 (Key reshuffling)**: Modulo hashing (`% len(ordered)`) remaps $\approx (N - 1)/N$ of all prefix keys whenever 1 replica scales out, drains, or restarts (e.g., 90% KV-cache invalidation at `N=10`). Ring Hash (Ketama virtual nodes) or Maglev lookup tables ([Envoy Load Balancers](https://envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers)) limit remapping to $\approx 1/N$.
  - **Flaw 2 (Unbounded hotspots & coarse prefix)**: `_extract_prefix` (`#L541-L545`) hashes only the first 256 characters of `system` (or first message), so all requests sharing a system prompt collide onto one replica until KEDA thresholds saturate, then herd onto a single `min()` replica. **Consistent Hashing with Bounded Loads (CHWBL)** caps every replica at $\lceil (1 + \epsilon) \cdot \text{avg\_load} \rceil$ and walks clockwise on the consistent hash ring to spill excess load while preserving $O(1/\epsilon^2)$ key stability ([Mirrokni et al., arXiv:1608.01350](https://arxiv.org/abs/1608.01350)). Production vLLM routers further score replicas using block-level KV prefix overlap minus a load penalty ([vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)).

---

## 6. Kubernetes Remediation & Anomaly Detection (`controller/self_healing.py`)

- **Correct as built — Guardrail state machine & actuator seam (`controller/self_healing.py#L44-L92`, `#L179-L268`)**:
  - `SelfHealingEngine` cleanly separates pure planning/guardrails from cluster side effects via `ClusterActuator` (`#L85-L92`), enforces `dry_run` (`#L232-L237`), deduplicates `(ActionKind, target)` (`#L208-L213`), applies per-target cooldowns (`#L215-L222`), caps concurrent node drains at 1 (`#L224-L230`), and escalates to `ActionKind.ESCALATE_TO_HUMAN` after `max_failures_before_page` (`#L253-L266`).

- **Production-gap / bug under multi-replica — `K8sActuatorAdapter` & `policy/v1` Eviction API (`controller/self_healing.py#L118-L140`, `#L175-L177`)**:
  - `K8sActuatorAdapter` (`#L118-L140`) is currently a thin callback wrapper without Kubernetes API calls. Per official Kubernetes node-maintenance guidelines ([Safely Drain a Node](https://kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/), [API-initiated Eviction](https://kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/)):
    1. A safe drain must first **cordon** the node (`node.spec.unschedulable = True`, applying `node.kubernetes.io/unschedulable:NoSchedule`) and then submit `policy/v1` `Eviction` subresource requests (`POST /api/v1/namespaces/{ns}/pods/{name}/eviction`) rather than raw `DELETE` pod calls so `PodDisruptionBudget` (`minAvailable` / `maxUnavailable`) rules are honored.
    2. When a PDB blocks eviction, the API server returns `429 Too Many Requests` ([API-initiated Eviction](https://kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/)); the actuator must retry with backoff and timeout, skip `DaemonSet` pods (e.g., NVIDIA device plugin / DCGM), and avoid blocking the synchronous `reconcile()` loop (`#L88`, `#L240`) while pods gracefully terminate.
    3. Nothing in `SelfHealingEngine` automatically invokes `complete_drain(guardrails, node)` (`#L175-L177`) upon eviction completion, and `GuardrailState` (`#L44-L53`) is process-local without K8s `Lease` leader election or shared persistent state.

- **Algorithmic upgrade — `compute_z_score` & univariate `PodTelemetry` (`controller/self_healing.py#L21-L42`, `#L142-L152`)**:
  - `compute_z_score` (`#L142-L152`) computes classical sample mean and standard deviation over `series[:-1]`. Classical mean/std have a 0% breakdown point: prior TTFT spikes or warm-up outliers in `series[:-1]` inflate both mean and variance, masking subsequent anomalies; conversely, a flat baseline (`std < 1e-6`, `#L150-L151`) returns `10.0` on negligible sub-millisecond jitter.
  - **Upgrade 1 (Univariate)**: Use the **Iglewicz–Hoaglin Modified Z-Score** $M_i = 0.6745(x_i - \tilde{x}) / \text{MAD}$ (50% breakdown point, recommended outlier threshold $|M_i| > 3.5$, floored by a minimum practical latency delta) per NIST guidance ([NIST/SEMATECH Detection of Outliers](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)).
  - **Upgrade 2 (Multivariate)**: `PodTelemetry` (`#L21-L33`) scores TTFT in isolation from `kv_cache_usage` and `gpu_temp_c` (except hard thresholds `#L194-L201`). **Isolation Forest** ([`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)) detects joint multivariate degradation across heavy-tailed TTFT, KV-cache saturation, queue depth, and thermal metrics without Gaussian assumptions.
