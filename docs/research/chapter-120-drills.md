# Chapter 120 Multi-Tenant Drills: Primary-Source Implementation Spec

Implementation specification for the three Chapter 120 multi-tenant drills mapped directly onto [`catalog/pool_catalog.py`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py), [`catalog/pools.yaml`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pools.yaml), [`gateway/request_control.py`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py), [`gateway/tenant_budget.py`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/tenant_budget.py), [`deploy/manifest_renderer.py`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py), and [`GLOSSARY.md`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/GLOSSARY.md).

---

## Executive Gap Summary

| Drill | Current Codebase State | Required Architectural Changes |
| :--- | :--- | :--- |
| **1. Dynamic Multi-Tenant LoRA (`--enable-lora`)** | `ModelPoolSpec` (`catalog/pool_catalog.py#L22-L43`) and `_build_vllm_args` (`#L115-L132`) model one base `model_id` per pool with no LoRA flags, no adapter ownership, and no adapter-aware CHWBL routing. | Add `LoraAdapterSpec` + pool LoRA engine flags in `catalog/pool_catalog.py`; enforce tenant ownership in `RequestControl`; salt CHWBL replica affinity with adapter name; prevent incompatible fallback to `pool-fallback`; wire `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` + `/v1/load_lora_adapter` behind `NetworkPolicy`. |
| **2. Prefix Cache Side-Channel Isolation** | `ChatCompletionRequest` sets `extra="allow"` (`gateway/request_control.py#L54`), allowing callers to supply or omit `cache_salt`; `_extract_prefix` (`#L859-L863`) hashes raw prompt prefixes without `tenant_id`; `DeployConfig.vllm_image` (`deploy/manifest_renderer.py#L18`) is pinned to `v0.7.3` (pre-`cache_salt`). | Upgrade `DeployConfig.vllm_image` to `>= v0.9.0` (V1 engine); strip caller-supplied `cache_salt`; inject server-HMAC `cache_salt` per tenant into upstream vLLM payloads; set `--prefix-caching-hash-algo sha256`; salt CHWBL key with `tenant_id`. |
| **3. Postgres Usage Rollups & Weight Prefetch DaemonSet** | `UsageEvent` (`gateway/request_control.py#L453-L460`) and `prefetch_targets` (`catalog/pool_catalog.py#L92-L99`) exist as unused hooks (`gateway/server.py#L89-L96` omits `on_usage`); `_vllm_deployment` mounts `/mnt/nvme/models` (`deploy/manifest_renderer.py#L136-L138`) without a prefetch `DaemonSet` or readiness barrier. | Emit `UsageEvent` to Redis Streams (`XADD ... MAXLEN ~`); consume via `XGROUP`/`XREADGROUP`/`XAUTOCLAIM` -> idempotent Postgres `INSERT ... ON CONFLICT DO UPDATE` with `trace_id` dedup -> `XACK`; add GPU `DaemonSet` using `s5cmd` + atomic staging rename + `.ready` sentinel + wait `initContainer`. |

---

## Drill 1: Multi-Tenant Dynamic LoRA Serving (`--enable-lora`)

### 1.1 Primary-Source vLLM Mechanics

- **Engine & CLI Flags** ([vLLM LoRA Docs](https://docs.vllm.ai/en/latest/models/lora.html), [vLLM Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html)):
  - `--enable-lora`: Allocates GPU LoRA slots and initializes the Punica multi-LoRA kernel dispatcher.
  - `--max-loras` (int, default `1`): Maximum concurrent LoRA adapters executed within a single batch (GPU active slot budget).
  - `--max-lora-rank` (int, default `16`): Maximum supported rank $r$; vLLM pre-allocates LoRA tensor buffers for this ceiling.
  - `--max-cpu-loras` (int | None, default `None`, constraint `max_cpu_loras >= max_loras`): CPU RAM cache tier holding evicted/warm adapters for fast GPU hot-swapping without disk/S3 reloads.
  - `--lora-dtype` (`"auto"`, `float16`, `bfloat16`, default `"auto"`): Matches base model dtype when `"auto"`.
  - `--fully-sharded-loras` (bool, default `False`): Under Tensor Parallelism (TP), shards both LoRA matrices $A$ and $B$ across TP ranks; faster for large TP, long sequences, or high rank.
  - `--lora-modules {name}={path}`: Registers static adapters at engine startup.
- **OpenAI `model` Parameter Resolution & Dynamic Loading** ([vLLM LoRA Docs](https://docs.vllm.ai/en/latest/models/lora.html)):
  - Callers target an adapter by passing `"model": "<lora_name>"` in `/v1/chat/completions` (or `/v1/completions`). vLLM maps `model` to the registered `lora_name` (also exposed via `GET /v1/models`).
  - Runtime adapter registration without pod restart requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` in the server environment, enabling `POST /v1/load_lora_adapter` (`{"lora_name": "...", "lora_path": "..."}`) and `POST /v1/unload_lora_adapter` (`{"lora_name": "..."}`).
  - **Security Constraint**: vLLM docs warn against exposing runtime LoRA endpoints to untrusted callers ([vLLM LoRA Docs](https://docs.vllm.ai/en/latest/models/lora.html)). In our cluster, `_inference_network_policy` ([`deploy/manifest_renderer.py#L199-L222`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L199-L222)) already restricts pod port `8000` ingress to `app: request-control` and `namespace: monitoring`, and `create_app` ([`gateway/request_control.py#L731-L737`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L731-L737)) exposes only `/v1/chat/completions` and `/metrics`, keeping `/v1/*_lora_adapter` private to the control plane.
- **Critical Interaction with Automatic Prefix Caching (`--enable-prefix-caching`)** ([vLLM APC Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)):
  - vLLM APC computes a chained block hash $\text{hash}(\text{parent\_hash} \parallel \text{token\_ids} \parallel \text{extra\_hashes})$, where `extra_hashes` explicitly includes the adapter's unique integer ID (`lora_int_id` / LoRA ID).
  - **Result**: APC and `--enable-lora` are **compatible and joint-managed**. KV blocks produced under LoRA A (`lora_int_id=1`) cannot match requests for LoRA B (`lora_int_id=2`) or the base model (`0`), preventing cross-adapter KV corruption or leakage while preserving prefix hits within the same adapter.
  - **Caveats**:
    1. Two tenants sharing the *same* base model or *same* shared LoRA adapter still require `cache_salt` (Drill 2) to prevent cross-tenant TTFT probing.
    2. Never reuse a `lora_name` / `lora_int_id` across different adapter revisions without unloading and bumping a versioned identifier (e.g., `tenant-a-sql@v2`), because block hashes depend on `lora_int_id`.

### 1.2 Concrete Codebase Mapping

1. **`catalog/pool_catalog.py` & `catalog/pools.yaml`**:
   - Add `LoraAdapterSpec(BaseModel)` with fields: `name: str`, `tenant_id: str | None = None` (`None` = shared adapter; non-null = private to one `tenant_id`), `s3_uri: str`, `rank: int = Field(default=16, ge=1)`, `dynamic: bool = True`.
   - Extend `ModelPoolSpec` ([`catalog/pool_catalog.py#L22-L50`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L22-L50)):
     - Add `enable_lora: bool = False`, `max_loras: int = Field(default=4, ge=1)`, `max_lora_rank: int = Field(default=16, ge=1)`, `max_cpu_loras: int = Field(default=16, ge=1)`, `lora_dtype: str = "auto"`, `fully_sharded_loras: bool = False`, `loras: list[LoraAdapterSpec] = Field(default_factory=list)`.
     - Enforce in `_validate_pool_invariants` (`#L44-L50`): `max_cpu_loras >= max_loras`, every `lora.rank <= max_lora_rank`, and `loras` non-empty implies `enable_lora=True`.
   - Extend `ModelPoolCatalog` ([`#L53-L113`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L53-L113)):
     - Index `self._loras: dict[str, tuple[str, LoraAdapterSpec]]` in `load_and_validate` (`#L135-L159`), rejecting duplicate names across `pools`, `_aliases`, and `_loras`.
     - Add `resolve_target(model_or_alias: str) -> ResolvedTarget(pool: ModelPoolSpec, lora: LoraAdapterSpec | None, engine_model: str)` (keep `resolve_alias` returning `ModelPoolSpec` for backward compatibility).
     - Update `_build_vllm_args(spec)` (`#L115-L132`): when `spec.enable_lora` is true, emit `--enable-lora`, `--max-loras`, `--max-lora-rank`, `--max-cpu-loras`, `--lora-dtype`, optional `--fully-sharded-loras`, and `--lora-modules` for static (`dynamic=False`) adapters pointing to `/models/loras/{spec.name}/{lora.name}`.
     - Update `render_targets()` (`#L92-L99`) `prefetch_targets` to include each pool's `loras` (`s3_uri` -> local target `/mnt/nvme/models/loras/{pool}/{lora.name}`) so the Drill 3 DaemonSet prefetches adapters alongside base weights.

2. **`gateway/request_control.py`**:
   - **Tenant Authorization**: In `chat_completions` (`#L581-L621`), resolve `target = deps.catalog.resolve_target(req.model)`. Immediately after `permit = deps.budget.authorize_and_reserve(...)` (`#L594-L598`) and **before** `deps.admission.acquire(...)` (`#L624`):
     - If `target.lora is not None` and `target.lora.tenant_id is not None` and `target.lora.tenant_id != permit.tenant_id`: call `deps.budget.reconcile(permit, 0)` and return `404` (`code="model_not_found"`) via `_error_response` (`#L431-L450`) so Tenant A cannot invoke or enumerate Tenant B's private LoRA.
   - **CHWBL Affinity Key & Fallback Guard**:
     - Update `_select_pool_and_replica` (`#L915-L931`) to build the affinity key from `(target.lora.name if target.lora else target.pool.model_id, prefix)` (plus tenant salt from Drill 2). Routing all requests for `lora_name` to the same consistent-hash ring neighborhood keeps that adapter hot in the target replica's `max_loras` GPU slots and `max_cpu_loras` CPU cache.
     - **Incompatible Base-Model Guard**: In `#L924-L930`, do **not** traverse `curr.fallback_pool` when `target.lora is not None`, because `pool-fallback` serves `Qwen/Qwen2.5-1.5B-Instruct` ([`catalog/pools.yaml#L37-L38`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pools.yaml#L37-L38)), which cannot execute a 7B LoRA adapter; return `503` if the primary pool has no healthy replicas.
   - **Upstream Dispatch & Dynamic Load**:
     - Pass `model_id=target.engine_model` (`lora.name` for LoRA requests, `target_pool.model_id` otherwise) into `deps.engine.complete(...)` (`#L688`) and `deps.engine.stream(...)` (`#L764`), and echo `target.engine_model` in the response `"model"` field (`#L714`, `#L797`, `#L821`).
     - In `VLLMAdapter` (`#L264-L390`), track per-replica loaded dynamic adapters in memory and on a `404` / first miss call `POST {replica.url}/v1/load_lora_adapter` with `{"lora_name": lora.name, "lora_path": f"/models/loras/{pool}/{lora.name}"}` (singleflight per `(replica_id, lora_name)` with `asyncio.Lock` to prevent thundering-herd loads).

3. **`deploy/manifest_renderer.py`**:
   - In `_vllm_deployment` (`#L73-L143`), when `spec.enable_lora` is true and any adapter has `dynamic=True`, inject `env: [{"name": "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "value": "true"}]` into the `vllm` container.

---

## Drill 2: Cross-Tenant Prefix Cache Side-Channel Isolation

### 2.1 Primary-Source Security & Engine Mechanics

- **The Timing Side-Channel Risk**:
  - Academic studies (*I Know What You Asked: Prompt Leakage via KV-Cache Sharing* [NDSS / Wu et al.](https://www.ndss-symposium.org/), *The Early Bird Catches the Leak* [arXiv](https://arxiv.org/)) and vLLM security advisory **CVE-2025-46570** ([vLLM Security Docs](https://docs.vllm.ai/en/latest/security/index.html)) demonstrate that global prefix sharing lets an adversary sharing a replica probe candidate prompt prefixes token-by-token and distinguish KV cache hits from misses via TTFT latency deltas.
  - Cryptographic per-principal salting (formalized in *KVGov*, [arXiv:2608.09225](https://arxiv.org/abs/2608.09225)) makes KV block hash chains disjoint across tenants ($\sigma_p = \text{HMAC}_K(\text{secret}, \text{tenant\_id})$) while keeping intra-tenant prefix reuse intact.
- **vLLM Engine-Level `cache_salt` & `--prefix-caching-hash-algo`**:
  - Introduced via vLLM RFC #16016 ([`vllm-project/vllm#16016`](https://github.com/vllm-project/vllm/issues/16016)) in **vLLM `v0.9.0`** to remediate **CVE-2025-46570** ([vLLM APC Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)):
    - Top-level request field `cache_salt: str` on `/v1/chat/completions` (and `/v1/completions`, `/v1/messages`) is folded into the **first KV block hash**. Requests with different `cache_salt` values never share KV blocks, even on identical prompts.
    - Supported **only on vLLM's V1 engine architecture** (not legacy Engine V0) ([vLLM APC Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)).
    - **Critical Version Gap in Repo**: [`deploy/manifest_renderer.py#L18`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L18) defaults `DeployConfig.vllm_image` to `"vllm/vllm-openai:v0.7.3"`, which **predates `v0.9.0`** and therefore ignores or rejects `cache_salt` / lacks CVE-2025-46570 mitigation. Upgrade `vllm_image` to `>= v0.9.0` (ideally `>= v0.30.0` if ever pairing with `LMCache-MP`, which patched `GHSA-2823-qmq8-rwvj` / `CVE-2026-105756` for `cache_salt` length/character validation: [GHSA-2823-qmq8-rwvj](https://github.com/advisories/GHSA-2823-qmq8-rwvj)).
  - `--prefix-caching-hash-algo`: Supports `sha256` (default) and `sha256_cbor` ([vLLM Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html)); eliminates non-cryptographic `xxhash` collision exposure.
- **Residual Boundary**: vLLM explicitly warns that `cache_salt` does **not** isolate scheduler queues, HBM eviction pressure, or container state ([vLLM Security Docs](https://docs.vllm.ai/en/latest/security/index.html), [vLLM APC Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)). Tenants requiring strict hardware/timing non-interference must use `IsolationMode.DEDICATED` pools ([`catalog/pool_catalog.py#L18`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L18)).

### 2.2 Routing-Level Salting vs. Engine-Level `cache_salt`

| Dimension | Routing-Level Salting (`gateway/request_control.py#_extract_prefix`) | Engine-Level `cache_salt` (`payload["cache_salt"]`) |
| :--- | :--- | :--- |
| **Mechanism** | Prepends `permit.tenant_id` to the CHWBL key before `_hash64` (`#L893`). | Folds gateway-derived HMAC salt into vLLM's first KV-block SHA-256 hash. |
| **Multi-Tenant Isolation** | **Insufficient alone**: `pool-8b` runs 1–4 replicas (`catalog/pools.yaml#L11-L12`), so $\ge 25\%$–$100\%$ of Tenant A & B requests land on the same pod and still share vLLM's global APC table. | **Complete prefix-block isolation** inside a shared replica; Tenant B gets zero KV reuse from Tenant A's cached prefix. |
| **Performance Effect** | Improves load balance across replicas when different tenants share identical boilerplate system prompts, while keeping same-tenant calls sticky. | Preserves full intra-tenant APC hits; trades cross-tenant sharing (which is unsafe anyway) for cryptographic tenant boundaries. |
| **Recommendation** | **Use both together**: Route with `tenant_id` + `prefix` AND inject server-HMAC `cache_salt` into the upstream payload. | Mandatory primary control on all `IsolationMode.SHARED` pools. |

### 2.3 Concrete Codebase Mapping

1. **`gateway/request_control.py`**:
   - **Fix `extra="allow"` Bypass (`#L54`)**: Because `ChatCompletionRequest` specifies `model_config = ConfigDict(extra="allow")` (`#L54`), `req.model_dump()` (`#L690`, `#L766`) currently forwards any caller-supplied `"cache_salt"` straight to vLLM! A malicious tenant could pass another tenant's salt or probe unsalted blocks.
   - **Gateway-Derived Salt**: Add helper `_derive_cache_salt(tenant_id: str, secret: bytes) -> str` computing `hmac.new(secret, tenant_id.encode("utf-8"), hashlib.sha256).hexdigest()` (64 lowercase hex chars — opaque, deterministic across all `RequestControl` pods, and free of special characters).
   - **Payload Sanitization & Injection**: In `chat_completions` / `VLLMAdapter`, build `upstream_payload = req.model_dump()`; always `upstream_payload.pop("cache_salt", None)`, and when `target_pool.isolate_prefix_cache` is true, set `upstream_payload["cache_salt"] = _derive_cache_salt(permit.tenant_id, deps.cache_salt_secret)`.
   - **CHWBL Salting (`#L859-L863`, `#L915-L931`)**: Pass `tenant_id=permit.tenant_id` into `_select_pool_and_replica` and form the CHWBL key as `f"{permit.tenant_id}\x00{target_model}\x00{_extract_prefix(messages)}"` when `isolate_prefix_cache` is enabled.
2. **`catalog/pool_catalog.py`**:
   - Add `isolate_prefix_cache: bool = True` and `prefix_caching_hash_algo: Literal["sha256", "sha256_cbor"] = "sha256"` to `ModelPoolSpec` (`#L22-L43`).
   - In `_build_vllm_args` (`#L130-L132`), when `spec.enable_prefix_caching` is true, emit `["--enable-prefix-caching", "--prefix-caching-hash-algo", spec.prefix_caching_hash_algo]`.
3. **`deploy/manifest_renderer.py` & `gateway/server.py`**:
   - Upgrade `DeployConfig.vllm_image` (`deploy/manifest_renderer.py#L18`) from `"vllm/vllm-openai:v0.7.3"` to `"vllm/vllm-openai:v0.9.0"` (or newer verified V1 release).
   - Inject `CACHE_SALT_SECRET` (from a Kubernetes `Secret`) into `_request_control_rollout` (`#L329-L341`) and read it in `gateway/server.py` (failing fast in production mode if unset rather than silently using a weak static default).

---

## Drill 3: Persistent Postgres Usage Rollups & Weight Prefetch DaemonSet

### 3.1 Part A — Redis Streams to Idempotent Postgres Billing Rollups

#### Primary-Source Mechanics
- **Redis Streams Producer & Consumer Group** ([Redis Streams](https://redis.io/docs/latest/develop/data-types/streams/)):
  - **Append (`XADD`)** ([`XADD`](https://redis.io/docs/latest/commands/xadd/)): `XADD usage:events MAXLEN ~ 100000 * trace_id <id> tenant_id <tid> pool <pool> prompt_tokens <p> completion_tokens <c> tokens <t> ts_ms <ms>`. The approximate `~` modifier avoids blocking macro-node rewrites on every append.
  - **Group Init (`XGROUP CREATE`)** ([`XGROUP CREATE`](https://redis.io/docs/latest/commands/xgroup-create/)): `XGROUP CREATE usage:events billing-rollup 0 MKSTREAM` (catch `BUSYGROUP` on restart).
  - **Read (`XREADGROUP`)** ([`XREADGROUP`](https://redis.io/docs/latest/commands/xreadgroup/)): `XREADGROUP GROUP billing-rollup <pod_id> COUNT 250 BLOCK 2000 STREAMS usage:events >` delivers new entries and records them in the group's Pending Entries List (PEL).
  - **Crash Recovery (`XAUTOCLAIM`)** ([`XAUTOCLAIM`](https://redis.io/docs/latest/commands/xautoclaim/)): On each poll tick (or before `XREADGROUP`), run `XAUTOCLAIM usage:events billing-rollup <pod_id> 60000 0-0 COUNT 250` to reclaim PEL entries unacknowledged for $>60\text{ s}$ after a worker crash.
  - **Ack (`XACK`)** ([`XACK`](https://redis.io/docs/latest/commands/xack/)): Call `XACK usage:events billing-rollup <id1> <id2> ...` **only after** Postgres commits the transaction.
- **PostgreSQL Idempotent Upsert + `trace_id` Deduplication** ([PostgreSQL `INSERT`](https://www.postgresql.org/docs/current/sql-insert.html)):
  - Because `XREADGROUP` + `XAUTOCLAIM` gives **at-least-once** delivery, a worker crashing between Postgres `COMMIT` and Redis `XACK` will replay the batch. Incrementing `ON CONFLICT DO UPDATE` without deduplication would double-bill!
  - Execute deduplication and hourly rollup atomically in **one Postgres transaction** using `INSERT ... ON CONFLICT DO NOTHING RETURNING trace_id` on a deduplication ledger (`usage_event_dedup`), then `INSERT ... ON CONFLICT (tenant_id, pool, bucket_hour) DO UPDATE` using `EXCLUDED`:

```sql
CREATE TABLE IF NOT EXISTS usage_event_dedup (
    trace_id   TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tenant_usage_hourly (
    tenant_id         TEXT NOT NULL,
    pool              TEXT NOT NULL,
    bucket_hour       TIMESTAMPTZ NOT NULL,
    prompt_tokens     BIGINT NOT NULL DEFAULT 0,
    completion_tokens BIGINT NOT NULL DEFAULT 0,
    total_tokens      BIGINT NOT NULL DEFAULT 0,
    request_count     BIGINT NOT NULL DEFAULT 0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, pool, bucket_hour)
);
```

```sql
WITH batch(trace_id, tenant_id, pool, bucket_hour, prompt_tokens, completion_tokens, total_tokens) AS (
    SELECT * FROM unnest(
        $1::text[], $2::text[], $3::text[], $4::timestamptz[],
        $5::bigint[], $6::bigint[], $7::bigint[]
    )
),
dedup AS (
    INSERT INTO usage_event_dedup (trace_id)
    SELECT trace_id FROM batch
    ON CONFLICT (trace_id) DO NOTHING
    RETURNING trace_id
),
agg AS (
    SELECT b.tenant_id, b.pool, date_trunc('hour', b.bucket_hour AT TIME ZONE 'UTC') AS bucket_hour,
           SUM(b.prompt_tokens) AS prompt_tokens,
           SUM(b.completion_tokens) AS completion_tokens,
           SUM(b.total_tokens) AS total_tokens,
           COUNT(*) AS request_count
    FROM batch b
    JOIN dedup d USING (trace_id)
    GROUP BY b.tenant_id, b.pool, date_trunc('hour', b.bucket_hour AT TIME ZONE 'UTC')
)
INSERT INTO tenant_usage_hourly (
    tenant_id, pool, bucket_hour, prompt_tokens, completion_tokens, total_tokens, request_count
)
SELECT tenant_id, pool, bucket_hour, prompt_tokens, completion_tokens, total_tokens, request_count
FROM agg
ON CONFLICT (tenant_id, pool, bucket_hour) DO UPDATE SET
    prompt_tokens     = tenant_usage_hourly.prompt_tokens + EXCLUDED.prompt_tokens,
    completion_tokens = tenant_usage_hourly.completion_tokens + EXCLUDED.completion_tokens,
    total_tokens      = tenant_usage_hourly.total_tokens + EXCLUDED.total_tokens,
    request_count     = tenant_usage_hourly.request_count + EXCLUDED.request_count,
    updated_at        = now();
```

*(Optionally add `lora_name TEXT NOT NULL DEFAULT ''` to the `tenant_usage_hourly` primary key `(tenant_id, pool, lora_name, bucket_hour)` to bill per-adapter usage from Drill 1.)*

#### Concrete Codebase Mapping
- **`gateway/request_control.py`**:
  - Extend `UsageEvent` (`#L453-L460`) with default-compatible fields: `prompt_tokens: int = 0`, `completion_tokens: int = 0`, `lora_name: str | None = None`, `recorded_at_ms: int = 0`, and populate them in `_record_usage_and_reconcile` (`#L474-L495`, `#L701-L708`, `#L843-L849`).
  - Keep `on_usage` non-blocking on the ASGI event loop (enqueue into a bounded `asyncio.Queue` drained by a background task in `lifespan` `#L543-L559`, or dispatch via non-blocking Redis pipeline, dropping/logging with a Prometheus counter only if Redis + local buffer saturate).
- **`gateway/tenant_budget.py` & `gateway/server.py`**:
  - Add a `RedisStreamUsageSink` (and standalone worker CLI/entrypoint) alongside `RedisPostgresAdapter` (`gateway/tenant_budget.py#L229`) and wire `on_usage=usage_sink.emit` in `create_default_app()` (`gateway/server.py#L89-L96`) when `REDIS_URL` is set.
  - Note: Also add an `on_usage` shutdown flush in `lifespan` (`gateway/request_control.py#L555-L559`) so graceful pod termination (`terminationGracePeriodSeconds`) drains buffered `UsageEvent`s before exit.

---

### 3.2 Part B — Weight Prefetch `DaemonSet` & Atomic S3-to-NVMe Sync

#### Primary-Source Mechanics
- **Kubernetes `DaemonSet` + `hostPath` + `initContainers`**:
  - A `DaemonSet` ([Kubernetes DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)) schedules one prefetch pod per GPU node matching Karpenter's `nvidia.com/gpu.present: "true"` label and `nvidia.com/gpu:NoSchedule` taint ([`deploy/manifest_renderer.py#L232-L236`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L232-L236)).
  - Sharing `/mnt/nvme/models` via `hostPath` (`type: DirectoryOrCreate`, [Kubernetes Volumes](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath)) lets multiple vLLM pods or restarts reuse warm node-local weights without re-downloading.
  - Pairing with a pod-level `initContainer` ([Kubernetes Init Containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/)) guarantees vLLM never launches before weights are complete on disk.
- **Karpenter Local NVMe Provisioning** ([Karpenter `EC2NodeClass`](https://karpenter.sh/docs/concepts/nodeclasses/)):
  - Currently `_karpenter_nodeclass` ([`deploy/manifest_renderer.py#L265-L286`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/manifest_renderer.py#L265-L286)) defines only a `100Gi` `gp3` EBS root volume and omits `instanceStorePolicy: RAID0`. Without `instanceStorePolicy: RAID0` (and mounting/symlinking the instance-store mount to `/mnt/nvme`), `g5`/`g6` local NVMe drives are ignored and `/mnt/nvme/models` writes onto slow `100Gi` EBS!
- **`s5cmd` vs `aws s3 cp`** ([`peak/s5cmd`](https://github.com/peak/s5cmd)):
  - `s5cmd` is a Go S3 client without Python GIL overhead that parallelizes multi-part range downloads across workers (`--concurrency 32`, wildcard `"s3://bucket/prefix/*"`), achieving up to ~12x higher throughput than `aws s3 cp` and saturating multi-Gbps EKS NICs + NVMe RAID0 arrays when paired with an S3 VPC Gateway Endpoint.
- **Atomic Swap / `.ready` Sentinel Protocol (Zero Partial Reads)**:
  1. Acquire a node-local advisory file lock (`flock /models/.locks/<slug>.lock`) shared across the `DaemonSet` and any fallback `initContainer`.
  2. Fast-path exit if `/models/<slug>/.ready` already exists and matches the expected manifest/ETag version.
  3. Download with `s5cmd cp --concurrency 32 "${PREFETCH_URI}/*" "/models/.staging/<slug>.$$/"` on the **same NVMe filesystem** (`/models`).
  4. Verify required files (`config.json` and `*.safetensors`), write `/models/.staging/<slug>.$$/.ready`, and atomically publish via POSIX `mv` (`rename(2)` on same mount):
     - If `/models/<slug>` does not exist: `mv "/models/.staging/<slug>.$$" "/models/<slug>"`.
     - If updating an existing directory: swap symlink `/models/<slug>` -> `/models/.revisions/<slug>-<rev>` atomically with `ln -sfn` + `mv -T`, never mutating a live directory in place.
  5. vLLM Deployment's `initContainer` polls `test -f /models/<slug>/.ready` (or invokes the same locked `s5cmd` script if the `DaemonSet` has not finished yet) before starting the main `vllm` container.
  6. Include an LRU / active-catalog GC pass in the `DaemonSet` loop because Kubernetes garbage collection does **not** manage `hostPath` disk usage ([Kubernetes Volumes](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath)).

#### Concrete Codebase Mapping
- **`catalog/pool_catalog.py`**:
  - Already computes `prefetch_targets` (`#L92-L99`) using `spec.prefetch_uri or f"s3://models/{spec.model_id}"` (note: [`deploy/terraform/main.tf#L63-L72`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/deploy/terraform/main.tf#L63-L72) provisions `aws_s3_bucket.model_weights`; pass bucket name via `DeployConfig.model_bucket_name`).
  - Add `use_local_nvme_weights: bool = False` (or `DeployConfig.use_prefetched_weights: bool`) so `_build_vllm_args` (`#L117-L120`) can pass `--model /models/<slug> --served-model-name <spec.model_id>` while preserving Prometheus `model_name` labels for KEDA (`#L155`, `#L178`)! Keeping `--served-model-name spec.model_id` is critical so KEDA's `model_name="Qwen/Qwen2.5-7B-Instruct"` queries do not break when `--model` switches to `/models/...`.
- **`deploy/manifest_renderer.py`**:
  - Add `_weight_prefetch_daemonset(prefetch_targets, cfg)` rendering `weight-prefetch-daemonset.yaml` (using `s5cmd`, `hostPath: /mnt/nvme/models`, GPU `nodeSelector` + `nvidia.com/gpu` toleration, and IRSA `serviceAccountName`).
  - Add a wait/sync `initContainer` inside `_vllm_deployment` (`#L90-L140`) checking `/models/<slug>/.ready`.
  - Add `"instanceStorePolicy": "RAID0"` to `_karpenter_nodeclass` (`#L265-L286`) so Karpenter aggregates local NVMe instance stores.

---

## Glossary Updates (`GLOSSARY.md`)

Add the following canonical domain terms to [`GLOSSARY.md`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/GLOSSARY.md):

- **LoRA Adapter**: A tenant-owned or shared low-rank weight delta served dynamically over a compatible base model pool (`--enable-lora`) and authorized per tenant before queue admission. _Avoid_: Fine-tuned model, plugin, sidecar weights.
- **Cache Salt**: A gateway-derived cryptographic per-tenant token injected into vLLM's first KV-block hash (`cache_salt`) to isolate automatic prefix cache namespaces against cross-tenant timing side channels. _Avoid_: Cache key, API key hash, session ID.
- **Usage Event**: An immutable post-completion or post-abort token accounting record (`trace_id`, `tenant_id`, `pool`, `prompt_tokens`, `completion_tokens`) appended to a durable stream for idempotent billing rollups. _Avoid_: Metric sample, invoice line, log entry.
- **Weight Prefetch DaemonSet**: A GPU-node Kubernetes `DaemonSet` that syncs model and LoRA weights from S3 to local NVMe (`hostPath`) via `s5cmd` with atomic `.ready` publication before inference pods start. _Avoid_: Model downloader job, sidecar cache.
