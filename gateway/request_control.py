from __future__ import annotations

import asyncio
import bisect
import hashlib
import heapq
import json
import math
import re
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

_VNODE_COUNT = 64
_PROM_METRIC_RE = re.compile(
    r"^(vllm:(?:num_requests_waiting|kv_cache_usage_perc))(?:\{[^}]*\})?\s+([0-9eE.+_-]+)$"
)


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=32768)
    stream: bool = False
    stream_options: dict[str, Any] | None = None


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


@dataclass(frozen=True)
class StreamChunk:
    """Single streamed token chunk emitted by an inference engine."""

    delta: str
    completion_tokens_so_far: int = 0
    prompt_tokens: int | None = None
    finished: bool = False
    role: str | None = None
    is_usage_only: bool = False


class InferenceEngine(Protocol):
    """Seam abstracting inference execution across vLLM and test adapters."""

    def list_replicas(self, pool: str) -> list[ReplicaEndpoint]: ...

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

    def list_replicas(self, pool: str) -> list[ReplicaEndpoint]:
        return list(self._replicas.get(pool, []))

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
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        )
        self._active_streams: dict[str, httpx.Response] = {}

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
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]
        return EngineResult(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
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
        body = {
            **payload,
            "model": model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        completion_count = 0
        async with self._client.stream(
            "POST",
            f"{replica.url}/v1/chat/completions",
            json=body,
            headers={"x-request-id": request_id},
        ) as resp:
            resp.raise_for_status()
            self._active_streams[request_id] = resp
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
                    if delta_text:
                        completion_count += 1
                    yield _make_stream_chunk(
                        delta=delta_text,
                        fallback_count=completion_count,
                        usage=usage if isinstance(usage, dict) else None,
                        finished=choice.get("finish_reason") is not None,
                        role=delta_obj.get("role"),
                    )
            finally:
                self._active_streams.pop(request_id, None)

    async def abort(self, *, replica: ReplicaEndpoint, request_id: str) -> None:
        """Close the active upstream HTTP stream so vLLM cancels generation."""
        _ = replica
        resp = self._active_streams.pop(request_id, None)
        if resp is not None:
            await resp.aclose()


def _make_stream_chunk(
    *,
    delta: str,
    fallback_count: int,
    usage: dict[str, Any] | None,
    finished: bool,
    role: str | None = None,
    is_usage_only: bool = False,
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
    )


def _estimate_prompt_tokens(messages: list[dict[str, Any]] | list[ChatMessage]) -> int:
    """Conservative UTF-8 byte-ratio + ChatML framing estimate for Qwen2.5 BPE."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg, ChatMessage) else str(msg.get("content", ""))
        byte_len = len(content.encode("utf-8"))
        total += max(1, math.ceil(byte_len / 3.0)) + 4
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


@dataclass
class _RuntimeDeps:
    catalog: ModelPoolCatalog
    budget: TenantBudget
    pool_health: PoolHealth
    engine: InferenceEngine
    admission: _PriorityAdmission
    metrics: MetricsRegistry | None = None
    on_usage: Callable[[UsageEvent], None] | None = None


def _record_usage_and_reconcile(
    *,
    deps: _RuntimeDeps,
    permit: Permit,
    pool_name: str,
    total_tokens: int,
    trace_id: str,
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
            )
        )


def _response_headers(pool_name: str, replica_id: str, trace_id: str) -> dict[str, str]:
    return {
        "x-served-pool": pool_name,
        "x-served-replica": replica_id,
        "x-trace-id": trace_id,
    }


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
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        scrape_fn = getattr(deps.engine, "scrape_metrics", None)
        task: asyncio.Task[None] | None = None
        if metrics_poll_interval_seconds is not None and callable(scrape_fn):

            async def _poll_loop() -> None:
                while True:
                    await scrape_fn()
                    await asyncio.sleep(metrics_poll_interval_seconds)

            task = asyncio.create_task(_poll_loop())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def metrics_endpoint(_request: Request) -> Response:
        active_metrics = deps.metrics or MetricsRegistry()
        return Response(
            content=active_metrics.render(),
            media_type=active_metrics.content_type,
        )

    async def chat_completions(request: Request) -> Response:
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        try:
            raw_body = await request.json()
            req = ChatCompletionRequest.model_validate(raw_body)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            return _error_response(
                message=f"Invalid request body: {exc}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
            )

        try:
            pool_spec = deps.catalog.resolve_alias(req.model)
        except KeyError as exc:
            return _error_response(
                message=str(exc),
                status_code=404,
                error_type="invalid_request_error",
                code="model_not_found",
            )

        est_prompt = _estimate_prompt_tokens(req.messages)
        est_total = est_prompt + req.max_tokens
        try:
            permit = deps.budget.authorize_and_reserve(
                _extract_api_key(request),
                pool_spec.name,
                est_total,
            )
        except AuthenticationError as exc:
            return _error_response(
                message=str(exc),
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )
        except BudgetExceededError as exc:
            if deps.metrics is not None:
                deps.metrics.rate_limited_total.labels(
                    tenant_id="unknown", reason="budget_exceeded"
                ).inc()
                deps.metrics.requests_total.labels(
                    pool=pool_spec.name, priority="unknown", status="429"
                ).inc()
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
            deps.budget.reconcile(permit, 0)
            if deps.metrics is not None:
                deps.metrics.rate_limited_total.labels(
                    tenant_id=permit.tenant_id, reason="queue_shed"
                ).inc()
                deps.metrics.requests_total.labels(
                    pool=pool_spec.name, priority=str(permit.priority), status="429"
                ).inc()
            return _error_response(
                message=str(exc),
                status_code=429,
                error_type="rate_limit_exceeded",
                code="queue_wait_exceeded",
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except BaseException:
            deps.budget.reconcile(permit, 0)
            raise
        if deps.metrics is not None:
            deps.metrics.queue_wait_seconds.labels(priority=str(permit.priority)).observe(
                max(0.0, time.perf_counter() - t_queue)
            )

        target_pool, replica = _select_pool_and_replica(
            deps=deps,
            initial_pool=pool_spec,
            messages=req.messages,
        )
        if replica is None:
            deps.admission.release()
            deps.budget.reconcile(permit, 0)
            if deps.metrics is not None:
                deps.metrics.requests_total.labels(
                    pool=pool_spec.name, priority=str(permit.priority), status="503"
                ).inc()
            return _error_response(
                message="No healthy replicas available",
                status_code=503,
                error_type="server_error",
                code="service_unavailable",
                headers={"Retry-After": "5"},
            )

        if deps.metrics is not None and target_pool.name != pool_spec.name:
            deps.metrics.fallback_total.labels(
                from_pool=pool_spec.name, to_pool=target_pool.name
            ).inc()

        t_req = time.perf_counter()
        if req.stream:
            return _streaming_chat_response(
                deps=deps,
                request=request,
                req=req,
                permit=permit,
                target_pool=target_pool,
                replica=replica,
                est_prompt=est_prompt,
                trace_id=trace_id,
                started_at=t_req,
            )

        try:
            result = await deps.engine.complete(
                replica=replica,
                model_id=target_pool.model_id,
                request_id=trace_id,
                payload=req.model_dump(),
            )
        finally:
            deps.admission.release()

        if deps.metrics is not None:
            deps.metrics.ttft_seconds.labels(
                pool=target_pool.name, priority=str(permit.priority)
            ).observe(max(0.0, time.perf_counter() - t_req))
            deps.metrics.requests_total.labels(
                pool=target_pool.name, priority=str(permit.priority), status="200"
            ).inc()

        total_tokens = result.prompt_tokens + result.completion_tokens
        _record_usage_and_reconcile(
            deps=deps,
            permit=permit,
            pool_name=target_pool.name,
            total_tokens=total_tokens,
            trace_id=trace_id,
        )
        return JSONResponse(
            {
                "id": f"chatcmpl-{trace_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": target_pool.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.content},
                        "finish_reason": "stop",
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
        ],
    )


def _streaming_chat_response(
    *,
    deps: _RuntimeDeps,
    request: Request,
    req: ChatCompletionRequest,
    permit: Permit,
    target_pool: ModelPoolSpec,
    replica: ReplicaEndpoint,
    est_prompt: int,
    trace_id: str,
    started_at: float = 0.0,
) -> StreamingResponse:
    include_usage = bool(
        req.stream_options and req.stream_options.get("include_usage", False)
    )
    created = int(time.time())

    async def event_stream() -> AsyncIterator[str]:
        completed = False
        emitted_tokens = 0
        actual_prompt_tokens: int | None = None
        recorded_ttft = False
        stream_iter = deps.engine.stream(
            replica=replica,
            model_id=target_pool.model_id,
            request_id=trace_id,
            payload=req.model_dump(),
        )
        try:
            async for chunk in stream_iter:
                if await request.is_disconnected():
                    break
                should_record_ttft = (
                    not recorded_ttft
                    and bool(chunk.delta)
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
                delta_obj: dict[str, Any] = {}
                if chunk.role is not None:
                    delta_obj["role"] = chunk.role
                if chunk.delta or chunk.role is not None:
                    delta_obj["content"] = chunk.delta
                payload = {
                    "id": f"chatcmpl-{trace_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": target_pool.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_obj,
                            "finish_reason": "stop" if chunk.finished else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
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
                        "model": target_pool.model_id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_used,
                            "completion_tokens": emitted_tokens,
                            "total_tokens": prompt_used + emitted_tokens,
                        },
                    }
                    yield f"data: {json.dumps(usage_payload)}\n\n"
                yield "data: [DONE]\n\n"
        finally:
            try:
                if not completed:
                    with anyio.CancelScope(shield=True):
                        aclose = getattr(stream_iter, "aclose", None)
                        if aclose is not None:
                            await aclose()
                        await deps.engine.abort(replica=replica, request_id=trace_id)
            finally:
                prompt_used = (
                    actual_prompt_tokens if actual_prompt_tokens is not None else est_prompt
                )
                _record_usage_and_reconcile(
                    deps=deps,
                    permit=permit,
                    pool_name=target_pool.name,
                    total_tokens=prompt_used + emitted_tokens,
                    trace_id=trace_id,
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
            return msg.content[:256]
    return messages[0].content[:256] if messages else ""


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _pick_replica(
    pool: ModelPoolSpec,
    replicas: list[ReplicaEndpoint],
    prefix: str,
    *,
    epsilon: float = 0.25,
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
            return candidate
        if len(visited) == len(by_id):
            break

    return min(
        replicas,
        key=lambda r: (r.requests_waiting, r.kv_cache_usage, r.replica_id),
    )


def _select_pool_and_replica(
    *,
    deps: _RuntimeDeps,
    initial_pool: ModelPoolSpec,
    messages: list[ChatMessage],
) -> tuple[ModelPoolSpec, ReplicaEndpoint | None]:
    prefix = _extract_prefix(messages)
    curr: ModelPoolSpec | None = initial_pool
    seen: set[str] = set()
    while curr is not None and curr.name not in seen:
        seen.add(curr.name)
        status = deps.pool_health.get_state(curr.name)
        replicas = deps.engine.list_replicas(curr.name)
        if status.state == HealthState.HEALTHY and replicas:
            return curr, _pick_replica(curr, replicas, prefix)
        curr = deps.catalog.pools.get(curr.fallback_pool) if curr.fallback_pool else None
    return initial_pool, None
