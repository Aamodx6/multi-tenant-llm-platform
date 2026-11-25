from __future__ import annotations

import asyncio
import bisect
import hashlib
import heapq
import hmac
import json
import logging
import math
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from catalog.pool_catalog import ModelPoolCatalog, ModelPoolSpec
from gateway.pool_health import HealthState, PoolHealth
from gateway.tenant_budget import (
    AuthenticationError,
    BudgetExceededError,
    Permit,
    PriorityClass,
    TenantBudget,
)
from observability.metrics import MetricsRegistry

_LOGGER = logging.getLogger(__name__)
_VNODE_COUNT = 64
_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")
_PROM_METRIC_RE = re.compile(
    r"^(vllm:(?:num_requests_waiting|kv_cache_usage_perc))(?:\{[^}]*\})?\s+([0-9eE.+_-]+)$"
)


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=32)
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload with strict field allowlist."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    cache_salt: str | None = None


@dataclass
class ReplicaEndpoint:
    """Inference replica metadata and live load signals."""

    replica_id: str
    url: str
    requests_waiting: int = 0
    kv_cache_usage: float = 0.0


@dataclass(frozen=True)
class EngineResult:
    """Completed inference output and actual token usage."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    replica_id: str
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class StreamChunk:
    """Single streamed token chunk emitted by an inference engine."""

    delta: str
    completion_tokens_so_far: int = 0
    prompt_tokens: int | None = None
    finished: bool = False
    role: str | None = None
    is_usage_only: bool = False
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


class InferenceEngine(Protocol):
    """Seam abstracting inference execution across vLLM and test adapters."""

    def list_replicas(self, pool: str) -> list[ReplicaEndpoint]: ...

    async def scrape_metrics(self) -> None: ...

    async def complete(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> EngineResult: ...

    def stream(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]: ...

    async def abort(self, *, replica: ReplicaEndpoint, request_id: str) -> None: ...

    async def ensure_lora_loaded(
        self, *, replica: ReplicaEndpoint, lora_name: str, lora_path: str
    ) -> None: ...

    def has_lora(self, replica_id: str, lora_name: str) -> bool: ...


class AdmissionPolicy(BaseModel):
    """In-process priority queue and load-shedding parameters."""

    max_in_flight: int = Field(default=64, ge=1)
    estimated_request_ms: int = Field(default=250, ge=1)
    free_wait_budget_ms: int = Field(default=500, ge=0)
    paid_wait_budget_ms: int = Field(default=5000, ge=0)


class _AdmissionShedError(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Admission queue wait budget exceeded.")
        self.retry_after_seconds = retry_after_seconds


class _PriorityAdmission:
    """Cancellation-safe in-process priority queue enforcing wait budgets without Redis."""

    def __init__(self, policy: AdmissionPolicy) -> None:
        self._policy = policy
        self._in_flight = 0
        self._seq = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    def _prune_done_waiters(self) -> None:
        while self._waiters and self._waiters[0][2].done():
            heapq.heappop(self._waiters)

    async def acquire(self, priority: PriorityClass) -> None:
        self._prune_done_waiters()
        if self._in_flight < self._policy.max_in_flight and not self._waiters:
            self._in_flight += 1
            return

        rank = 0 if priority == PriorityClass.PAID else 1
        ahead = sum(1 for r, _, f in self._waiters if not f.done() and r <= rank)
        expected_wait_ms = (1 + ahead) * self._policy.estimated_request_ms
        budget_ms = (
            self._policy.paid_wait_budget_ms
            if priority == PriorityClass.PAID
            else self._policy.free_wait_budget_ms
        )
        if expected_wait_ms > budget_ms:
            raise _AdmissionShedError(max(1, expected_wait_ms // 1000 + 1))

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (rank, self._seq, fut))
        try:
            await fut
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                fut.cancel()
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._in_flight = max(0, self._in_flight - 1)


class FakeEngineAdapter:
    """In-memory InferenceEngine adapter for deterministic RequestControl tests."""

    def __init__(
        self,
        replicas: dict[str, list[ReplicaEndpoint]],
        *,
        completion_tokens: int = 10,
        stream_chunks: list[str] | None = None,
        complete_gate: asyncio.Event | None = None,
        on_start: asyncio.Event | None = None,
        stream_gate: asyncio.Event | None = None,
    ) -> None:
        self._replicas = {k: list(v) for k, v in replicas.items()}
        self._completion_tokens = completion_tokens
        self._stream_chunks = stream_chunks or ["hello", " world"]
        self._complete_gate = complete_gate
        self._on_start = on_start
        self.stream_gate = stream_gate
        self.dispatched: list[tuple[str, str, str]] = []
        self.aborted: list[tuple[str, str]] = []
        self.loaded_loras: set[tuple[str, str]] = set()

    def list_replicas(self, pool: str) -> list[ReplicaEndpoint]:
        return list(self._replicas.get(pool, []))

    async def scrape_metrics(self) -> None:
        return None

    async def complete(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> EngineResult:
        self.dispatched.append((replica.replica_id, model_id, request_id))
        if self._on_start is not None:
            self._on_start.set()
        if self._complete_gate is not None:
            await self._complete_gate.wait()
        prompt_tokens = _estimate_prompt_tokens(payload.get("messages", []))
        return EngineResult(
            content="ok",
            prompt_tokens=prompt_tokens,
            completion_tokens=self._completion_tokens,
            replica_id=replica.replica_id,
        )

    async def stream(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        self.dispatched.append((replica.replica_id, model_id, request_id))
        prompt_tokens = _estimate_prompt_tokens(payload.get("messages", []))
        for idx, token in enumerate(self._stream_chunks, start=1):
            if idx > 1 and self.stream_gate is not None:
                await self.stream_gate.wait()
            is_last = idx == len(self._stream_chunks)
            yield StreamChunk(
                delta=token,
                completion_tokens_so_far=idx,
                prompt_tokens=prompt_tokens if is_last else None,
                finished=is_last,
            )

    async def abort(self, *, replica: ReplicaEndpoint, request_id: str) -> None:
        self.aborted.append((replica.replica_id, request_id))

    async def ensure_lora_loaded(
        self, *, replica: ReplicaEndpoint, lora_name: str, lora_path: str
    ) -> None:
        _ = lora_path
        self.loaded_loras.add((replica.replica_id, lora_name))

    def has_lora(self, replica_id: str, lora_name: str) -> bool:
        return (replica_id, lora_name) in self.loaded_loras


class VLLMAdapter:
    """Production InferenceEngine adapter calling vLLM OpenAI endpoints over HTTP."""

    def __init__(
        self,
        replicas: dict[str, list[ReplicaEndpoint]],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._replicas = {k: list(v) for k, v in replicas.items()}
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=5.0)
        )
        self._active_streams: dict[str, list[httpx.Response]] = {}
        self._loaded_loras: set[tuple[str, str]] = set()
        self._lora_paths: dict[tuple[str, str], str] = {}
        self._lora_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def ensure_lora_loaded(
        self, *, replica: ReplicaEndpoint, lora_name: str, lora_path: str
    ) -> None:
        """Load a dynamic LoRA adapter onto a vLLM replica once using singleflight locking."""
        key = (replica.replica_id, lora_name)
        self._lora_paths[key] = lora_path
        if key in self._loaded_loras:
            return
        lock = self._lora_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._loaded_loras:
                return
            resp = await self._client.post(
                f"{replica.url}/v1/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
            )
            resp.raise_for_status()
            self._loaded_loras.add(key)

    async def _reload_lora_on_404(self, replica: ReplicaEndpoint, model_id: str) -> bool:
        key = (replica.replica_id, model_id)
        lora_path = self._lora_paths.get(key)
        if lora_path is None:
            return False
        self._loaded_loras.discard(key)
        await self.ensure_lora_loaded(replica=replica, lora_name=model_id, lora_path=lora_path)
        return True

    def has_lora(self, replica_id: str, lora_name: str) -> bool:
        return (replica_id, lora_name) in self._loaded_loras

    def list_replicas(self, pool: str) -> list[ReplicaEndpoint]:
        return list(self._replicas.get(pool, []))

    async def scrape_metrics(self) -> None:
        """Poll vLLM /metrics on all replicas to refresh queue and KV-cache load."""
        for pool_replicas in self._replicas.values():
            for replica in pool_replicas:
                try:
                    resp = await self._client.get(f"{replica.url}/metrics")
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue
                for line in resp.text.splitlines():
                    match = _PROM_METRIC_RE.match(line.strip())
                    if not match:
                        continue
                    metric, raw_val = match.group(1), match.group(2)
                    val = float(raw_val)
                    if metric == "vllm:num_requests_waiting":
                        replica.requests_waiting = int(val)
                    elif metric == "vllm:kv_cache_usage_perc":
                        replica.kv_cache_usage = val

    async def complete(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> EngineResult:
        body = {**payload, "model": model_id, "stream": False}
        resp = await self._client.post(
            f"{replica.url}/v1/chat/completions",
            json=body,
            headers={"x-request-id": request_id},
        )
        if resp.status_code == 404 and await self._reload_lora_on_404(replica, model_id):
            resp = await self._client.post(
                f"{replica.url}/v1/chat/completions",
                json=body,
                headers={"x-request-id": request_id},
            )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        return EngineResult(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            replica_id=replica.replica_id,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            tool_calls=message.get("tool_calls"),
            reasoning_content=message.get("reasoning_content"),
        )

    async def stream(
        self,
        *,
        replica: ReplicaEndpoint,
        model_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        body = {
            **payload,
            "model": model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        completion_count = 0
        for attempt in range(2):
            async with self._client.stream(
                "POST",
                f"{replica.url}/v1/chat/completions",
                json=body,
                headers={"x-request-id": request_id},
            ) as resp:
                if (
                    attempt == 0
                    and resp.status_code == 404
                    and await self._reload_lora_on_404(replica, model_id)
                ):
                    continue
                resp.raise_for_status()
                active = self._active_streams.setdefault(request_id, [])
                active.append(resp)
                try:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        event = json.loads(raw)
                        usage = event.get("usage")
                        choices = event.get("choices") or []
                        if not choices:
                            if isinstance(usage, dict):
                                yield _make_stream_chunk(
                                    delta="",
                                    fallback_count=completion_count,
                                    usage=usage,
                                    finished=True,
                                    is_usage_only=True,
                                )
                            continue
                        choice = choices[0]
                        delta_obj = choice.get("delta") or {}
                        delta_text = str(delta_obj.get("content") or "")
                        reasoning = delta_obj.get("reasoning_content")
                        tool_calls = delta_obj.get("tool_calls")
                        if delta_text or reasoning or tool_calls:
                            completion_count += 1
                        finish_reason = choice.get("finish_reason")
                        yield _make_stream_chunk(
                            delta=delta_text,
                            fallback_count=completion_count,
                            usage=usage if isinstance(usage, dict) else None,
                            finished=finish_reason is not None,
                            finish_reason=str(finish_reason) if finish_reason is not None else None,
                            role=delta_obj.get("role"),
                            tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                            reasoning_content=str(reasoning) if reasoning is not None else None,
                        )
                finally:
                    entries = self._active_streams.get(request_id)
                    if entries is not None:
                        with suppress(ValueError):
                            entries.remove(resp)
                        if not entries:
                            self._active_streams.pop(request_id, None)
                return

    async def abort(self, *, replica: ReplicaEndpoint, request_id: str) -> None:
        """Close the active upstream HTTP stream so vLLM cancels generation."""
        _ = replica
        resps = self._active_streams.pop(request_id, [])
        for resp in resps:
            await resp.aclose()


def _make_stream_chunk(
    *,
    delta: str,
    fallback_count: int,
    usage: dict[str, Any] | None,
    finished: bool,
    finish_reason: str | None = None,
    role: str | None = None,
    is_usage_only: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
) -> StreamChunk:
    comp = int(usage.get("completion_tokens", fallback_count)) if usage else fallback_count
    prompt = int(usage["prompt_tokens"]) if usage and "prompt_tokens" in usage else None
    return StreamChunk(
        delta=delta,
        completion_tokens_so_far=comp,
        prompt_tokens=prompt,
        finished=finished,
        role=role,
        is_usage_only=is_usage_only,
        finish_reason=finish_reason or ("stop" if finished else None),
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )


def _content_bytes(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return len(json.dumps(content, separators=(",", ":")).encode("utf-8"))


def _estimate_prompt_tokens(
    messages: list[dict[str, Any]] | list[ChatMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> int:
    """Conservative UTF-8 byte-ratio + ChatML framing estimate for Qwen2.5 BPE."""
    total = 0
    for msg in messages:
        if isinstance(msg, ChatMessage):
            byte_len = _content_bytes(msg.content)
            if msg.name:
                byte_len += len(msg.name.encode("utf-8"))
            if msg.tool_calls:
                byte_len += len(json.dumps(msg.tool_calls, separators=(",", ":")).encode("utf-8"))
        else:
            byte_len = _content_bytes(msg.get("content", ""))
            if msg.get("name"):
                byte_len += len(str(msg["name"]).encode("utf-8"))
            if msg.get("tool_calls"):
                byte_len += len(
                    json.dumps(msg["tool_calls"], separators=(",", ":")).encode("utf-8")
                )
        total += (max(1, math.ceil(byte_len / 3.0)) if byte_len > 0 else 1) + 4
    if tools:
        tools_bytes = len(json.dumps(tools, separators=(",", ":")).encode("utf-8"))
        total += max(1, math.ceil(tools_bytes / 3.0))
    if response_format:
        rf_bytes = len(json.dumps(response_format, separators=(",", ":")).encode("utf-8"))
        total += max(1, math.ceil(rf_bytes / 3.0))
    return max(1, total)


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
        status_code=status_code,
        headers=headers,
    )


@dataclass(frozen=True)
class UsageEvent:
    """Metered token consumption emitted per completed or aborted request."""

    trace_id: str
    tenant_id: str
    pool: str
    tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    lora_name: str | None = None
    recorded_at_ms: int = 0
    event_id: str = ""


@dataclass
class _RuntimeDeps:
    catalog: ModelPoolCatalog
    budget: TenantBudget
    pool_health: PoolHealth
    engine: InferenceEngine
    admission: _PriorityAdmission
    metrics: MetricsRegistry | None = None
    on_usage: Callable[[UsageEvent], None] | None = None
    cache_salt_secret: bytes = b""


def _record_usage_and_reconcile(
    *,
    deps: _RuntimeDeps,
    permit: Permit,
    pool_name: str,
    total_tokens: int,
    trace_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    lora_name: str | None = None,
) -> None:
    deps.budget.reconcile(permit, total_tokens)
    if deps.metrics is not None and total_tokens > 0:
        deps.metrics.tokens_total.labels(tenant_id=permit.tenant_id, pool=pool_name).inc(
            total_tokens
        )
    if deps.on_usage is not None:
        deps.on_usage(
            UsageEvent(
                trace_id=trace_id,
                tenant_id=permit.tenant_id,
                pool=pool_name,
                tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                lora_name=lora_name,
                recorded_at_ms=int(time.time() * 1000),
                event_id=uuid.uuid4().hex,
            )
        )


def _response_headers(pool_name: str, replica_id: str, trace_id: str) -> dict[str, str]:
    return {
        "x-served-pool": pool_name,
        "x-served-replica": replica_id,
        "x-trace-id": trace_id,
    }


def _record_rate_limit(
    deps: _RuntimeDeps, *, pool: str, tenant_id: str, priority: str, reason: str
) -> None:
    if deps.metrics is None:
        return
    deps.metrics.rate_limited_total.labels(tenant_id=tenant_id, reason=reason).inc()
    deps.metrics.requests_total.labels(pool=pool, priority=priority, status="429").inc()


def _record_status(deps: _RuntimeDeps, *, pool: str, priority: str, status: str) -> None:
    if deps.metrics is not None:
        deps.metrics.requests_total.labels(pool=pool, priority=priority, status=status).inc()


def _build_upstream_payload(
    *,
    req: ChatCompletionRequest,
    pool: ModelPoolSpec,
    tenant_id: str,
    secret: bytes,
    effective_max_tokens: int,
) -> dict[str, Any]:
    """Strip caller cache_salt, enforce bounded max_tokens, and inject HMAC tenant salt."""
    payload = req.model_dump(exclude_none=True)
    payload.pop("cache_salt", None)
    payload.pop("max_completion_tokens", None)
    payload["max_tokens"] = effective_max_tokens
    if pool.enable_prefix_caching and pool.isolate_prefix_cache:
        payload["cache_salt"] = hmac.new(
            secret, tenant_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()
    return payload


def _extract_bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


async def _fail_with_503(
    *,
    deps: _RuntimeDeps,
    permit: Permit,
    pool_name: str,
    message: str,
    code: str,
) -> JSONResponse:
    deps.admission.release()
    await _reconcile_async(deps, permit, 0)
    _record_status(deps, pool=pool_name, priority=str(permit.priority), status="503")
    return _error_response(
        message=message,
        status_code=503,
        error_type="server_error",
        code=code,
        headers={"Retry-After": "5"},
    )


async def _shielded_permit_cleanup(
    deps: _RuntimeDeps, permit: Permit, *, release_admission: bool
) -> None:
    if release_admission:
        deps.admission.release()
    with anyio.CancelScope(shield=True):
        await _reconcile_async(deps, permit, 0)


def _sanitize_trace_id(raw: str | None) -> str:
    if raw and _TRACE_ID_RE.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


def create_app(
    *,
    catalog: ModelPoolCatalog,
    budget: TenantBudget,
    pool_health: PoolHealth,
    engine: InferenceEngine,
    admission_policy: AdmissionPolicy | None = None,
    metrics: MetricsRegistry | None = None,
    metrics_poll_interval_seconds: float | None = None,
    on_usage: Callable[[UsageEvent], None] | None = None,
    on_usage_flush: Callable[[], Any] | None = None,
    cache_salt_secret: bytes | None = None,
    max_request_bytes: int = 4 * 1024 * 1024,
    metrics_auth_token: str | None = None,
) -> Starlette:
    """Create the unified RequestControl ASGI application."""
    deps = _RuntimeDeps(
        catalog=catalog,
        budget=budget,
        pool_health=pool_health,
        engine=engine,
        admission=_PriorityAdmission(admission_policy or AdmissionPolicy()),
        metrics=metrics,
        on_usage=on_usage,
        cache_salt_secret=cache_salt_secret or secrets.token_bytes(32),
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if metrics_poll_interval_seconds is not None:

            async def _poll_loop() -> None:
                while True:
                    await deps.engine.scrape_metrics()
                    await asyncio.sleep(metrics_poll_interval_seconds)

            task = asyncio.create_task(_poll_loop())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if on_usage_flush is not None:
                try:
                    on_usage_flush()
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning("Failed to flush buffered usage events on shutdown: %s", exc)

    async def metrics_endpoint(request: Request) -> Response:
        if metrics_auth_token is not None:
            presented = _extract_bearer_token(request)
            if not hmac.compare_digest(presented, metrics_auth_token):
                return _error_response(
                    message="Unauthorized metrics access.",
                    status_code=401,
                    error_type="authentication_error",
                    code="unauthorized_metrics",
                )
        active_metrics = deps.metrics or MetricsRegistry()
        return Response(
            content=active_metrics.render(),
            media_type=active_metrics.content_type,
        )

    async def healthz_endpoint(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz_endpoint(_request: Request) -> Response:
        has_replicas = any(deps.engine.list_replicas(name) for name in deps.catalog.pools)
        if not has_replicas:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    async def chat_completions(request: Request) -> Response:
        trace_id = _sanitize_trace_id(request.headers.get("x-trace-id"))
        internal_request_id = uuid.uuid4().hex
        raw_cl = request.headers.get("content-length")
        if raw_cl is not None:
            try:
                if int(raw_cl) > max_request_bytes:
                    return _error_response(
                        message=f"Request body exceeds maximum size of {max_request_bytes} bytes.",
                        status_code=413,
                        error_type="invalid_request_error",
                        code="payload_too_large",
                    )
            except ValueError:
                pass

        raw_bytes = await request.body()
        if len(raw_bytes) > max_request_bytes:
            return _error_response(
                message=f"Request body exceeds maximum size of {max_request_bytes} bytes.",
                status_code=413,
                error_type="invalid_request_error",
                code="payload_too_large",
            )

        try:
            raw_body = json.loads(raw_bytes.decode("utf-8"))
            req = ChatCompletionRequest.model_validate(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, ValidationError) as exc:
            return _error_response(
                message=f"Invalid request body: {exc}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
            )

        try:
            resolved = deps.catalog.resolve_target(req.model)
        except KeyError as exc:
            return _error_response(
                message=str(exc),
                status_code=404,
                error_type="invalid_request_error",
                code="model_not_found",
            )
        pool_spec = resolved.pool
        lora_spec = resolved.lora

        api_key = _extract_api_key(request)
        try:
            tenant = await anyio.to_thread.run_sync(deps.budget.authenticate, api_key)
        except AuthenticationError as exc:
            return _error_response(
                message=str(exc),
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )

        if (
            lora_spec is not None
            and lora_spec.tenant_id is not None
            and tenant.tenant_id != lora_spec.tenant_id
        ):
            _record_status(
                deps, pool=pool_spec.name, priority=str(tenant.priority), status="404"
            )
            return _error_response(
                message=f"Unknown model alias or pool: {req.model}",
                status_code=404,
                error_type="invalid_request_error",
                code="model_not_found",
            )

        est_prompt = _estimate_prompt_tokens(
            req.messages, tools=req.tools, response_format=req.response_format
        )
        effective_max_tokens = min(
            pool_spec.max_model_len,
            req.max_completion_tokens or req.max_tokens or 256,
        )
        est_total = est_prompt + effective_max_tokens
        try:
            permit = await anyio.to_thread.run_sync(
                deps.budget.reserve, tenant, pool_spec.name, est_total
            )
        except BudgetExceededError as exc:
            _record_rate_limit(
                deps,
                pool=pool_spec.name,
                tenant_id=exc.tenant_id,
                priority=str(exc.priority) if exc.priority is not None else "unknown",
                reason="budget_exceeded",
            )
            return _error_response(
                message=str(exc),
                status_code=429,
                error_type="rate_limit_exceeded",
                code="rate_limit_exceeded",
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )

        t_queue = time.perf_counter()
        try:
            await deps.admission.acquire(permit.priority)
        except _AdmissionShedError as exc:
            await _reconcile_async(deps, permit, 0)
            _record_rate_limit(
                deps,
                pool=pool_spec.name,
                tenant_id=permit.tenant_id,
                priority=str(permit.priority),
                reason="queue_shed",
            )
            return _error_response(
                message=str(exc),
                status_code=429,
                error_type="rate_limit_exceeded",
                code="queue_wait_exceeded",
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except BaseException:
            await _shielded_permit_cleanup(deps, permit, release_admission=False)
            raise
        if deps.metrics is not None:
            deps.metrics.queue_wait_seconds.labels(priority=str(permit.priority)).observe(
                max(0.0, time.perf_counter() - t_queue)
            )

        target_pool, replica = await _select_pool_and_replica_async(
            deps=deps,
            initial_pool=pool_spec,
            messages=req.messages,
            allow_fallback=lora_spec is None,
            lora_name=lora_spec.name if lora_spec is not None else None,
            tenant_id=permit.tenant_id,
        )
        if replica is None:
            return await _fail_with_503(
                deps=deps,
                permit=permit,
                pool_name=pool_spec.name,
                message="No healthy replicas available",
                code="service_unavailable",
            )

        if deps.metrics is not None and target_pool.name != pool_spec.name:
            deps.metrics.fallback_total.labels(
                from_pool=pool_spec.name, to_pool=target_pool.name
            ).inc()

        served_model = lora_spec.name if lora_spec is not None else target_pool.model_id
        if lora_spec is not None and lora_spec.dynamic:
            try:
                await deps.engine.ensure_lora_loaded(
                    replica=replica,
                    lora_name=lora_spec.name,
                    lora_path=f"/models/loras/{target_pool.name}/{lora_spec.name}",
                )
            except httpx.HTTPError as exc:
                return await _fail_with_503(
                    deps=deps,
                    permit=permit,
                    pool_name=target_pool.name,
                    message=f"Failed to load dynamic LoRA adapter: {exc}",
                    code="lora_load_failed",
                )
            except BaseException:
                await _shielded_permit_cleanup(deps, permit, release_admission=True)
                raise

        upstream_payload = _build_upstream_payload(
            req=req,
            pool=target_pool,
            tenant_id=permit.tenant_id,
            secret=deps.cache_salt_secret,
            effective_max_tokens=effective_max_tokens,
        )

        t_req = time.perf_counter()
        if req.stream:
            return await _open_streaming_chat_response(
                deps=deps,
                request=request,
                req=req,
                permit=permit,
                target_pool=target_pool,
                served_model=served_model,
                lora_name=lora_spec.name if lora_spec is not None else None,
                upstream_payload=upstream_payload,
                replica=replica,
                est_prompt=est_prompt,
                trace_id=trace_id,
                internal_request_id=internal_request_id,
                started_at=t_req,
            )

        try:
            result = await deps.engine.complete(
                replica=replica,
                model_id=served_model,
                request_id=internal_request_id,
                payload=upstream_payload,
            )
        except httpx.HTTPError as exc:
            return await _fail_with_503(
                deps=deps,
                permit=permit,
                pool_name=target_pool.name,
                message=f"Inference upstream unavailable: {exc}",
                code="service_unavailable",
            )
        except BaseException:
            await _shielded_permit_cleanup(deps, permit, release_admission=True)
            raise
        deps.admission.release()

        if deps.metrics is not None:
            deps.metrics.e2el_seconds.labels(
                pool=target_pool.name, priority=str(permit.priority)
            ).observe(max(0.0, time.perf_counter() - t_req))
        _record_status(deps, pool=target_pool.name, priority=str(permit.priority), status="200")

        total_tokens = result.prompt_tokens + result.completion_tokens
        await anyio.to_thread.run_sync(
            lambda: _record_usage_and_reconcile(
                deps=deps,
                permit=permit,
                pool_name=target_pool.name,
                total_tokens=total_tokens,
                trace_id=trace_id,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                lora_name=lora_spec.name if lora_spec is not None else None,
            )
        )
        msg_payload: dict[str, Any] = {
            "role": "assistant",
            "content": result.content if result.tool_calls is None or result.content else None,
        }
        if result.tool_calls is None:
            msg_payload["content"] = result.content
        else:
            msg_payload["tool_calls"] = result.tool_calls
        if result.reasoning_content is not None:
            msg_payload["reasoning_content"] = result.reasoning_content

        return JSONResponse(
            {
                "id": f"chatcmpl-{trace_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_payload,
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": total_tokens,
                },
            },
            headers=_response_headers(target_pool.name, result.replica_id, trace_id),
        )

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/metrics", metrics_endpoint, methods=["GET"]),
            Route("/healthz", healthz_endpoint, methods=["GET"]),
            Route("/readyz", readyz_endpoint, methods=["GET"]),
        ],
    )


async def _reconcile_async(deps: _RuntimeDeps, permit: Permit, actual_tokens: int) -> None:
    await anyio.to_thread.run_sync(deps.budget.reconcile, permit, actual_tokens)


def _format_sse_chunk(
    *,
    chunk: StreamChunk,
    trace_id: str,
    created: int,
    served_model: str,
) -> str:
    delta_obj: dict[str, Any] = {}
    if chunk.role is not None:
        delta_obj["role"] = chunk.role
    if chunk.delta or chunk.role is not None:
        delta_obj["content"] = chunk.delta
    if chunk.reasoning_content is not None:
        delta_obj["reasoning_content"] = chunk.reasoning_content
    if chunk.tool_calls is not None:
        delta_obj["tool_calls"] = chunk.tool_calls
    finish_reason = chunk.finish_reason if chunk.finished else None
    if chunk.finished and finish_reason is None:
        finish_reason = "stop"
    payload = {
        "id": f"chatcmpl-{trace_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model,
        "choices": [
            {
                "index": 0,
                "delta": delta_obj,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _open_streaming_chat_response(
    *,
    deps: _RuntimeDeps,
    request: Request,
    req: ChatCompletionRequest,
    permit: Permit,
    target_pool: ModelPoolSpec,
    served_model: str,
    lora_name: str | None = None,
    upstream_payload: dict[str, Any],
    replica: ReplicaEndpoint,
    est_prompt: int,
    trace_id: str,
    internal_request_id: str,
    started_at: float = 0.0,
) -> Response:
    include_usage = bool(
        req.stream_options and req.stream_options.get("include_usage", False)
    )
    created = int(time.time())
    stream_iter = deps.engine.stream(
        replica=replica,
        model_id=served_model,
        request_id=internal_request_id,
        payload=upstream_payload,
    )

    first_chunk: StreamChunk | None = None
    try:
        first_chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except httpx.HTTPError as exc:
        with anyio.CancelScope(shield=True):
            aclose = getattr(stream_iter, "aclose", None)
            if aclose is not None:
                await aclose()
        return await _fail_with_503(
            deps=deps,
            permit=permit,
            pool_name=target_pool.name,
            message=f"Inference upstream unavailable: {exc}",
            code="service_unavailable",
        )
    except BaseException:
        with anyio.CancelScope(shield=True):
            aclose = getattr(stream_iter, "aclose", None)
            if aclose is not None:
                await aclose()
        await _shielded_permit_cleanup(deps, permit, release_admission=True)
        raise

    async def event_stream() -> AsyncIterator[str]:
        completed = False
        upstream_failed = False
        emitted_tokens = 0
        actual_prompt_tokens: int | None = None
        recorded_ttft = False

        async def _iter_chunks() -> AsyncIterator[StreamChunk]:
            if first_chunk is not None:
                yield first_chunk
            async for nxt in stream_iter:
                yield nxt

        try:
            async for chunk in _iter_chunks():
                if await request.is_disconnected():
                    break
                has_output = bool(chunk.delta or chunk.reasoning_content or chunk.tool_calls)
                should_record_ttft = (
                    not recorded_ttft
                    and has_output
                    and deps.metrics is not None
                    and started_at > 0
                )
                if should_record_ttft and deps.metrics is not None:
                    deps.metrics.ttft_seconds.labels(
                        pool=target_pool.name, priority=str(permit.priority)
                    ).observe(max(0.0, time.perf_counter() - started_at))
                    recorded_ttft = True
                emitted_tokens = chunk.completion_tokens_so_far
                if chunk.prompt_tokens is not None:
                    actual_prompt_tokens = chunk.prompt_tokens
                if chunk.is_usage_only:
                    continue
                yield _format_sse_chunk(
                    chunk=chunk,
                    trace_id=trace_id,
                    created=created,
                    served_model=served_model,
                )
            else:
                completed = True
                if deps.metrics is not None:
                    deps.metrics.requests_total.labels(
                        pool=target_pool.name, priority=str(permit.priority), status="200"
                    ).inc()
                if include_usage:
                    prompt_used = (
                        actual_prompt_tokens if actual_prompt_tokens is not None else est_prompt
                    )
                    usage_payload = {
                        "id": f"chatcmpl-{trace_id}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": served_model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_used,
                            "completion_tokens": emitted_tokens,
                            "total_tokens": prompt_used + emitted_tokens,
                        },
                    }
                    yield f"data: {json.dumps(usage_payload)}\n\n"
                yield "data: [DONE]\n\n"
        except httpx.HTTPError as exc:
            upstream_failed = True
            _record_status(deps, pool=target_pool.name, priority=str(permit.priority), status="503")
            err_payload = {
                "error": {
                    "message": f"Upstream inference error mid-stream: {exc}",
                    "type": "server_error",
                    "param": None,
                    "code": "service_unavailable",
                }
            }
            yield f"data: {json.dumps(err_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            try:
                if not completed:
                    with anyio.CancelScope(shield=True):
                        aclose = getattr(stream_iter, "aclose", None)
                        if aclose is not None:
                            await aclose()
                        await deps.engine.abort(replica=replica, request_id=internal_request_id)
            finally:
                with anyio.CancelScope(shield=True):
                    if upstream_failed and emitted_tokens == 0:
                        await _reconcile_async(deps, permit, 0)
                    else:
                        prompt_used = (
                            actual_prompt_tokens
                            if actual_prompt_tokens is not None
                            else est_prompt
                        )
                        await anyio.to_thread.run_sync(
                            lambda: _record_usage_and_reconcile(
                                deps=deps,
                                permit=permit,
                                pool_name=target_pool.name,
                                total_tokens=prompt_used + emitted_tokens,
                                trace_id=trace_id,
                                prompt_tokens=prompt_used,
                                completion_tokens=emitted_tokens,
                                lora_name=lora_name,
                            )
                        )
                    deps.admission.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_response_headers(target_pool.name, replica.replica_id, trace_id),
    )


def _extract_prefix(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.role == "system":
            text_val = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            return text_val[:256]
    if not messages:
        return ""
    first = messages[0].content
    text_val = first if isinstance(first, str) else json.dumps(first)
    return text_val[:256]


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _pick_replica(
    pool: ModelPoolSpec,
    replicas: list[ReplicaEndpoint],
    prefix: str,
    *,
    epsilon: float = 0.25,
    has_lora_fn: Callable[[str, str], bool] | None = None,
    lora_name: str | None = None,
) -> ReplicaEndpoint:
    """Consistent Hashing with Bounded Loads (CHWBL) across replica virtual nodes."""
    by_id = {r.replica_id: r for r in replicas}
    ring: list[tuple[int, str]] = []
    for rep_id in by_id:
        for vnode in range(_VNODE_COUNT):
            ring.append((_hash64(f"{rep_id}#{vnode}"), rep_id))
    ring.sort()

    avg_load = sum(r.requests_waiting for r in replicas) / len(replicas)
    waiting_cap = (
        min(pool.keda_waiting_threshold, max(1, math.ceil((1.0 + epsilon) * avg_load)))
        if avg_load > 0.0
        else pool.keda_waiting_threshold
    )

    ring_hashes = [h for h, _ in ring]
    start_idx = bisect.bisect_left(ring_hashes, _hash64(prefix))
    visited: set[str] = set()
    eligible: list[ReplicaEndpoint] = []
    for offset in range(len(ring)):
        rep_id = ring[(start_idx + offset) % len(ring)][1]
        if rep_id in visited:
            continue
        visited.add(rep_id)
        candidate = by_id[rep_id]
        if (
            candidate.requests_waiting <= waiting_cap
            and candidate.kv_cache_usage <= pool.keda_kv_cache_threshold
        ):
            if lora_name is None or has_lora_fn is None or has_lora_fn(rep_id, lora_name):
                return candidate
            eligible.append(candidate)
        if len(visited) == len(by_id):
            break

    if eligible:
        return eligible[0]

    return min(
        replicas,
        key=lambda r: (r.requests_waiting, r.kv_cache_usage, r.replica_id),
    )


async def _select_pool_and_replica_async(
    *,
    deps: _RuntimeDeps,
    initial_pool: ModelPoolSpec,
    messages: list[ChatMessage],
    allow_fallback: bool = True,
    lora_name: str | None = None,
    tenant_id: str = "",
) -> tuple[ModelPoolSpec, ReplicaEndpoint | None]:
    prefix = _extract_prefix(messages)
    base_key = f"lora:{lora_name}:{prefix}" if lora_name is not None else prefix
    curr: ModelPoolSpec | None = initial_pool
    seen: set[str] = set()
    while curr is not None and curr.name not in seen:
        seen.add(curr.name)
        status = await anyio.to_thread.run_sync(deps.pool_health.get_state, curr.name)
        replicas = deps.engine.list_replicas(curr.name)
        if status.state == HealthState.HEALTHY and replicas:
            routing_key = (
                f"{tenant_id}:{base_key}" if curr.isolate_prefix_cache and tenant_id else base_key
            )
            return curr, _pick_replica(
                curr,
                replicas,
                routing_key,
                has_lora_fn=deps.engine.has_lora,
                lora_name=lora_name,
            )
        if not allow_fallback:
            break
        curr = deps.catalog.pools.get(curr.fallback_pool) if curr.fallback_pool else None
    return initial_pool, None
