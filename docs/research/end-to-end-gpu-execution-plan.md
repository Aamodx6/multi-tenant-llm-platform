# End-to-End GPU Execution Plan: `VLLMAdapter`, `/bench`, `/deploy`, `/observability`, and Timed Spot-GPU Runbook

Research blueprint cross-checking the current repository against upstream primary sources for vLLM, GuideLLM, KEDA, NVIDIA DCGM Exporter, Karpenter, and AWS/RunPod GPU pricing.

---

## 1. Concrete Mismatches Found in Current Codebase

| Area | File & Lines | Current Code Behavior | Upstream Reality & Required Fix |
| --- | --- | --- | --- |
| **Abort endpoint** | [`gateway/request_control.py#L290-L294`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L290-L294) | `VLLMAdapter.abort` sends `POST {replica.url}/v1/abort` with `{"request_id": request_id}`. | Standard vLLM OpenAI endpoints have **no `/v1/abort` route**. `/v1/chat/completions` is wrapped with `@with_cancellation` / `listen_for_disconnect(request)` ([vllm-project/vllm](https://github.com/vllm-project/vllm)), so closing the HTTP stream connection aborts generation and frees KV-cache blocks (note: custom `--middleware` can break `is_disconnected()`, vLLM `#10087`). Scale-out control endpoint `/abort_requests` (or `/inference/v1/abort_requests`) exists only for `--enable-scale-out` / `--tokens-only` setups ([vLLM Security Docs](https://docs.vllm.ai/en/latest/security/index.html)). **Fix:** Track active `httpx` stream contexts/tasks per `request_id` in `VLLMAdapter` and close/cancel the stream on abort; remove `/v1/abort`. |
| **Streaming SSE & `include_usage`** | [`gateway/request_control.py#L268-L289`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L268-L289), [`#L495-L532`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L495-L532) | `stream()` omits `stream_options`, yields raw JSON `line[6:]` as `StreamChunk.delta`, counts every SSE line as 1 token, and `_streaming_chat_response()` wraps `chunk.delta` inside another `delta.content` JSON object while reconciling `TenantBudget` with `est_prompt + emitted_tokens`. | vLLM `OpenAIServingChat` requires `"stream_options": {"include_usage": True}` (unless server sets `enable_force_include_usage`) to emit the trailing `choices: []` chunk carrying authoritative `usage` ([vLLM Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html)). **Fix:** Parse each SSE JSON chunk for `choices[0].delta.content` and trailing `usage` (`prompt_tokens`, `completion_tokens`), pass through cleanly, and reconcile `TenantBudget` from final `usage` (falling back to `est_prompt + emitted_tokens` only on disconnect/abort). |
| **Request payload passthrough** | [`gateway/request_control.py#L37-L44`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L37-L44) | `ChatCompletionRequest` only declares `model`, `messages`, `max_tokens`, `stream` (Pydantic drops `temperature`, `top_p`, `stop`, etc.). | Add `model_config = ConfigDict(extra="allow")` or explicit OpenAI sampling fields (`temperature`, `top_p`, `stop`, `stream_options`) so `vllm bench serve` / `guidellm` parameters pass through unmodified. |
| **Live replica metrics polling** | [`gateway/request_control.py#L46-L54`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L46-L54), [`#L548-L560`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/gateway/request_control.py#L548-L560) | `ReplicaEndpoint.requests_waiting` and `kv_cache_usage` default to `0`/`0.0`; nothing scrapes vLLM `/metrics` in production. | Add a background `/metrics` scraper in `VLLMAdapter` parsing `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` so bounded-load routing uses live engine pressure. |
| **Deprecated vLLM metric names** | [`docs/Multi-Tenant LLM Serving Platform Architecture.md#L56`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/docs/Multi-Tenant%20LLM%20Serving%20Platform%20Architecture.md#L56) | Refers generically to TPOT and KV-cache usage. | In current vLLM, `vllm:kv_cache_usage_perc` replaces legacy `vllm:gpu_cache_usage_perc`, and `vllm:time_per_output_token_seconds` is deprecated in favor of `vllm:inter_token_latency_seconds` ([vLLM Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)). |
| **AWQ checkpoint vs flag** | [`catalog/pool_catalog.py#L28`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/catalog/pool_catalog.py#L28), [`tests/test_request_control.py#L27`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/tests/test_request_control.py#L27) | `ModelPoolSpec` keeps `model_id: Qwen/Qwen2.5-7B-Instruct` with `quantization: str = "fp16"`; `render_targets()` emits no vLLM engine flags. | `--quantization awq` cannot quantize `Qwen/Qwen2.5-7B-Instruct` on the fly; it requires an AWQ checkpoint such as `Qwen/Qwen2.5-7B-Instruct-AWQ` ([Hugging Face Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)). Extend `render_targets()` to emit vLLM CLI args (`--dtype`, `--quantization`, `--max-model-len`, `--gpu-memory-utilization`, `--enable-prefix-caching`). |
| **DCGM default CSV & v4.9+ renames** | [`controller/self_healing.py#L21-L34`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L21-L34), [`#L191-L195`](file:///c:/Users/aamod/OneDrive/Desktop/Apps/Multi%20Tenant/controller/self_healing.py#L191-L195) | `PodTelemetry` requires `gpu_temp_c`, `xid_errors`, `ecc_errors`, `kv_cache_usage`, `oom_restarts`, `ttft_ms_series`. | Upstream `default-counters.csv` enables `DCGM_FI_DEV_GPU_TEMP` by default, while `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_XID_ERRORS`, and ECC counters often require an explicit custom CSV/ConfigMap; moreover DCGM Exporter v4.9.0+ renamed canonical metrics/units (`_RATIO`, `_BYTES`, `_CELSIUS`, `DCGM_FI_DEV_XID_ERROR`) ([NVIDIA DCGM Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)). Pin DCGM Exporter image + ship an explicit `counters.csv`. |

---

## 2. vLLM Serving & `VLLMAdapter` (`/gateway`)

### 2.1 Streaming Usage & Request Cancellation
- **Usage reporting:** Send `"stream_options": {"include_usage": true}` on every streaming `/v1/chat/completions` request. vLLM's `OpenAIServingChat` appends a trailing chunk with `choices: []` and `usage: {prompt_tokens, completion_tokens, total_tokens}` before `data: [DONE]` ([vLLM Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html)).
- **Cancellation/abort:**
  1. In `VLLMAdapter.stream()`, register an `asyncio.Event` / active `httpx.Response` handle keyed by `request_id` (and pass `x-request-id: {request_id}` for trace correlation).
  2. In `_streaming_chat_response()` (`finally:` when `not completed`), close the `VLLMAdapter.stream()` async generator via `aclose()` and call `abort(replica=..., request_id=...)` to close the underlying `httpx` TCP stream immediately. Closing the HTTP connection triggers vLLM's `@with_cancellation` / `listen_for_disconnect` guard to run `engine.abort(request_id)` and release KV-cache blocks ([vllm-project/vllm](https://github.com/vllm-project/vllm)).
  3. **Do not** pass custom `--middleware` flags to `vllm serve` (known to break `request.is_disconnected()` in Starlette, vLLM `#10087`), and use separate `httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)` rather than a flat `60.0s`.

### 2.2 vLLM Prometheus `/metrics` Names & Labels
All core metrics are exposed on `:8000/metrics` with the `model_name` label (populated from `--served-model-name` or `--model`) ([vLLM Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)):

| Signal | Exact vLLM Metric | Type | Notes |
| --- | --- | --- | --- |
| Queue depth | `vllm:num_requests_waiting` | Gauge | Waiting + preempted requests; primary KEDA & bounded-load signal |
| Active batch | `vllm:num_requests_running` | Gauge | Actively executing sequences |
| KV cache pressure | `vllm:kv_cache_usage_perc` | Gauge (`0.0–1.0`) | Replaces legacy `vllm:gpu_cache_usage_perc` |
| TTFT latency | `vllm:time_to_first_token_seconds` | Histogram | SLO metric (`histogram_quantile(0.95, ...)`) |
| Inter-token latency | `vllm:inter_token_latency_seconds` | Histogram | Replaces deprecated `vllm:time_per_output_token_seconds` |
| Prefix cache hits/queries | `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries` | Counter | Verify exact `_total` suffix against live `/metrics` at server boot |
| Preemptions & aborts | `vllm:num_preemptions_total`, `vllm:request_success_total{finished_reason="abort"}` | Counter | Confirms disconnect-abort & KV pressure behavior |

### 2.3 Target Model Launch Flags (24 GB L4 / A10G vs 6 GB RTX 4050)
Verified against [vLLM Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html) and [Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ):

- **Pool A — FP16/BF16 (`pool-8b` on 24 GB L4/A10G):**
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching --port 8000
  ```
- **Pool A-AWQ (`pool-8b-awq` on 24 GB L4/A10G):**
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --served-model-name Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching --port 8001
  ```
  *(vLLM auto-detects AWQ from `quantization_config`; passing `--quantization awq` explicitly documents intent and fails fast on non-AWQ weights.)*
- **Pool Fallback (`pool-fallback`, runs on 24 GB cloud GPU or 6 GB local RTX 4050 smoke test):**
  ```bash
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
    --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8002
  ```

---

## 3. Benchmarking & Cost/1M Tokens (`/bench`)

### 3.1 Official Tools (`vllm bench serve` & `guidellm`)
- **`vllm bench serve`** (wraps `benchmark_serving.py`, installed via `pip install vllm[bench]`; [vLLM Serving Benchmark](https://docs.vllm.ai/en/latest/dev/benchmarking/serving.html)):
  ```bash
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://127.0.0.1:8080 \
    --model qwen-7b \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --request-rate 8 \
    --num-prompts 200 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --goodput ttft:1000 tpot:50
  ```
  Exports request throughput (req/s), output token throughput (tok/s), total token throughput (tok/s), and p50/p95/p99 TTFT, TPOT, ITL, E2EL.
- **`guidellm`** ([vllm-project/guidellm](https://github.com/vllm-project/guidellm)): Runs automated rate sweeps (`--profile kind=sweep,sweep_size=6`) to find the exact max sustainable RPS/tok/s under the `p95 TTFT <= 1000 ms` SLO and exports structured JSON/CSV/HTML reports (`GenerativeBenchmarksReport`).

### 3.2 Cost per 1M Tokens Formula
Using the spot/pod hourly GPU price $P_{\text{hr}}$ (`$/hr`) and steady-state measured throughput $T_{\text{tok/s}}$ (`tok/s`) at the highest load tier meeting `p95 TTFT <= 1000 ms`:

$$\text{Cost per 1M tokens (\$)} = \frac{P_{\text{hr}}}{3600 \times T_{\text{tok/s}}} \times 1{,}000{,}000$$

Report **both** Output-only `$/1M output tokens` ($T_{\text{output}}$) and Blended `$/1M total tokens` ($T_{\text{total}} = T_{\text{input}} + T_{\text{output}}$), plus goodput-adjusted cost if requests breach SLO.

### 3.3 Clean A/B Matrix (1–2 Hour Spot-GPU Window)
Keep prompt distribution, `max_tokens=256`, and `request-rate` steps identical across arms; warm up with 20 requests before each measured run:
1. **A/B #1 — Prefix-Cache Routing ON vs Round-Robin** (target: `RequestControl` gateway -> 2 vLLM replicas or 2 logical workers sharing GPU / multi-tenant shared system prompt of ~1,500 tokens + 128-token unique suffix):
   - **Arm A:** `RequestControl` SHA-256 prefix-affinity bounded-load router + `--enable-prefix-caching`.
   - **Arm B:** Round-robin across replicas (same `--enable-prefix-caching` server, or prefix caching disabled baseline).
   - **Capture:** `p95 TTFT` reduction, `vllm:prefix_cache_hits / vllm:prefix_cache_queries`, output tok/s.
2. **A/B #2 — `Qwen2.5-7B-Instruct` (BF16/FP16) vs `Qwen2.5-7B-Instruct-AWQ`**:
   - **Capture:** `p95 TTFT`, `p95 ITL/TPOT`, peak sustainable output & total tok/s under `p95 TTFT <= 1000 ms`, `$/1M tokens`, and `vllm:kv_cache_usage_perc` saturation concurrency.

---

## 4. Kubernetes Deployment, Autoscaling & Observability (`/deploy`, `/observability`)

### 4.1 KEDA `ScaledObject` (`keda.sh/v1alpha1`)
Per [KEDA ScaledObject Spec](https://keda.sh/docs/reference/scaledobject-spec/) and [KEDA Prometheus Scaler](https://keda.sh/docs/scalers/prometheus/), Prometheus `query` must return a single scalar/vector numeric value; `minReplicaCount: 0` uses `activationThreshold` to prevent premature wakeups:
- **Shared pool (`pool-8b`, `minReplicaCount: 1`, `maxReplicaCount: 4`):**
  - `pollingInterval: 15`, `cooldownPeriod: 300`
  - Trigger 1 (`vllm:num_requests_waiting`): `query: sum(vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"})`, `threshold: "4"` (from `ModelPoolCatalog.keda_waiting_threshold`).
  - Trigger 2 (`vllm:kv_cache_usage_perc`): `query: avg(vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct"})`, `threshold: "0.80"` (from `keda_kv_cache_threshold`).
- **Fallback / Dedicated pools (`pool-fallback`, `minReplicaCount: 0`, `maxReplicaCount: 2`):**
  - `minReplicaCount: 0`, `cooldownPeriod: 300`, `activationThreshold: "1"`, `threshold: "6"`.

### 4.2 NVIDIA DCGM Exporter -> `controller/self_healing.py` (`PodTelemetry`)
Per [NVIDIA DCGM Exporter Docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) and [NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter):
- Ship an explicit `counters.csv` ConfigMap (do **not** rely on `default-counters.csv`, which omits several error/utilization fields) and pin the exporter tag (account for v4.9.0+ canonical renames):

| `PodTelemetry` Field | DCGM / K8s Source Metric (Pre-4.9 / v4.9+ Canonical) | Normalization |
| --- | --- | --- |
| `gpu_temp_c` | `DCGM_FI_DEV_GPU_TEMP` / `DCGM_FI_DEV_GPU_TEMP_CELSIUS` | °C (`>= 85.0` -> `CORDON_AND_DRAIN`) |
| `xid_errors` | `DCGM_FI_DEV_XID_ERRORS` / `DCGM_FI_DEV_XID_ERROR` | Count `> 0` -> `CORDON_AND_DRAIN` |
| `ecc_errors` | `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` (+ `SBE` aggregate) | Uncorrectable DBE `> 0` -> `CORDON_AND_DRAIN` |
| GPU util & VRAM (Grafana) | `DCGM_FI_DEV_GPU_UTIL` (`_RATIO`), `DCGM_FI_DEV_FB_USED` (`_BYTES`) | Convert ratio/bytes consistently |
| `kv_cache_usage` | `vllm:kv_cache_usage_perc` | `0.0–1.0` (`>= 0.95` -> degrade + restart) |
| `oom_restarts` | `kube_pod_container_status_restarts_total` + `OOMKilled` (`kube-state-metrics`) | Delta restarts `> 0` |
| `ttft_ms_series` | `vllm:time_to_first_token_seconds` (or gateway per-request TTFT ring buffer) | Rolling window in ms for `compute_z_score` |

### 4.3 Karpenter & Cloud Spot GPU Comparison
Per [Karpenter NodePools](https://karpenter.sh/docs/concepts/nodepools), [Karpenter NodeClasses](https://karpenter.sh/docs/concepts/nodeclasses), [AWS G6](https://aws.amazon.com/ec2/instance-types/g6/), [AWS G5](https://aws.amazon.com/ec2/instance-types/g5/), [AWS EC2 Pricing](https://aws.amazon.com/ec2/spot/pricing/), and [RunPod Pricing](https://runpod.io/pricing):

| Option | GPU (VRAM) | vCPU / RAM / Local NVMe | Hourly Rate (Ref) | Best Role |
| --- | --- | --- | --- | --- |
| **AWS `g6.xlarge`** | 1× NVIDIA L4 (24 GiB, Ada) | 4 vCPU / 16 GiB / EBS | ~$0.805/hr on-demand; ~$0.16–$0.53/hr spot | EKS Karpenter `NodePool` (`["spot", "on-demand"]`, `instance-family: ["g6", "g5"]`) for KEDA + Karpenter burst scale timing |
| **AWS `g5.xlarge`** | 1× NVIDIA A10G (24 GiB, Ampere) | 4 vCPU / 16 GiB / 250 GB NVMe | ~$1.006/hr on-demand; ~$0.35–$0.63/hr spot | EKS Karpenter diversified spot fallback pool + NVMe weight cache |
| **RunPod L4 Pod** | 1× NVIDIA L4 (24 GB) | Cloud Pod (per-second billing, $0 egress) | ~$0.44/hr Community, ~$0.69/hr Secure | Fastest 1–2 hr timed benchmark & fault-injection run (~$1 total) without EKS cluster setup overhead |

---

## 5. Minimal Timed-Session Runbook

### Phase A — Local Pre-Flight (Free, on 6 GB RTX 4050 + Local `kind`/Docker)
1. Implement `VLLMAdapter` fixes (disconnect-abort, `stream_options.include_usage`, SSE delta parsing, live `/metrics` poller) and unit-test with mocked `httpx` SSE streams (`pytest`, `mypy`, `ruff`).
2. Validate `/deploy` manifests (KEDA `ScaledObject`, Prometheus scrape configs, DCGM `counters.csv` ConfigMap, Grafana dashboards, Karpenter `NodePool`/`EC2NodeClass` YAML) against `kind` + `kubeconform` / `helm template`.
3. Smoke-test `RequestControl` + `vllm serve Qwen/Qwen2.5-1.5B-Instruct` (or AWQ) + `/bench` scripts locally on the 6 GB RTX 4050 so zero rented GPU minutes are spent debugging scripts.

### Phase B — Rented GPU Window (1–2 Hours)
- **Option 1 (RunPod L4 Pod, ~1.5 hrs, <$1.20):** Captures real **p95 TTFT**, **output & total throughput**, **prefix-cache A/B**, **FP16 vs AWQ A/B**, **$/1M tokens**, and **single-node pod restart / `PoolHealth` fallback MTTR** (note: single-node Pod cannot measure multi-node EKS Karpenter cold-start).
- **Option 2 (EKS + Karpenter `g6.xlarge`/`g5.xlarge` spot, ~2 hrs):** Captures all metrics above **plus** real **burst scale time** (KEDA trigger -> Karpenter `NodeClaim` EC2 spot boot -> NVIDIA device plugin -> weight pull -> `vllm` `/health` ready) and **node-drain MTTR** (`SelfHealingEngine` -> `K8sActuatorAdapter.cordon_and_drain`).

| Minute | Step | Recorded Output Metric |
| --- | --- | --- |
| `0–15` | Boot instance, pre-pull `Qwen2.5-7B-Instruct`, `Qwen2.5-7B-Instruct-AWQ`, `Qwen2.5-1.5B-Instruct`; verify `/metrics` field names | Cold model load time (s); verified metric names |
| `15–40` | Run A/B #1 (Prefix-cache ON vs Round-Robin) via `/bench` through `RequestControl` | `p95 TTFT`, `prefix_cache_hits/queries`, `req/s` |
| `40–70` | Run A/B #2 (`Qwen2.5-7B-Instruct` BF16 vs AWQ) with `guidellm` sweep + `vllm bench serve` | `p95 TTFT`, `p95 ITL/TPOT`, max SLO `tok/s`, `$/1M tokens` |
| `70–95` | Fault injection: inject `kv_cache >= 0.95` / synthetic XID or kill vLLM pod while `pool-fallback` serves traffic | **MTTR** (detection -> `PoolHealth` degraded fallback -> pod ready); **burst scale time** (if on EKS/Karpenter) |
| `95–105` | Export JSON/CSV bench reports, Prometheus/Grafana snapshots, and `SelfHealingEngine` audit logs; terminate GPU | Artifacts committed under `bench/results/` |
