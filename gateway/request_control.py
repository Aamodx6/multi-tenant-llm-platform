from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field
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


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=32768)
    stream: bool = False


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
    finished: bool = False


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
    """In-process priority queue enforcing wait budgets without Redis."""

    def __init__(self, policy: AdmissionPolicy) -> None:
        self._policy = policy
        self._in_flight = 0
        self._seq = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    async def acquire(self, priority: PriorityClass) -> None:
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
        for idx, token in enumerate(self._stream_chunks, start=1):
            if idx > 1 and self.stream_gate is not None:
                await self.stream_gate.wait()
            yield StreamChunk(
                delta=token,
                completion_tokens_so_far=idx,
                finished=(idx == len(self._stream_chunks)),
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
        self._client = client or httpx.AsyncClient(timeout=60.0)

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
        body = {**payload, "model": model_id, "stream": True}
        async with self._client.stream(
            "POST",
            f"{replica.url}/v1/chat/completions",
            json=body,
            headers={"x-request-id": request_id},
        ) as resp:
            resp.raise_for_status()
            count = 0
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    count += 1
                    yield StreamChunk(delta=line[6:], completion_tokens_so_far=count)

    async def abort(self, *, replica: ReplicaEndpoint, request_id: str) -> None:
        await self._client.post(
            f"{replica.url}/v1/abort",
            json={"request_id": request_id},
        )


def _estimate_prompt_tokens(messages: list[dict[str, Any]] | list[ChatMessage]) -> int:
    total_words = 0
    for msg in messages:
        content = msg.content if isinstance(msg, ChatMessage) else str(msg.get("content", ""))
        total_words += max(1, len(content.split()))
    return max(1, total_words)


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


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
    on_usage: Callable[[UsageEvent], None] | None = None,
) -> Starlette:
    """Create the unified RequestControl ASGI application."""
    deps = _RuntimeDeps(
        catalog=catalog,
        budget=budget,
        pool_health=pool_health,
        engine=engine,
        admission=_PriorityAdmission(admission_policy or AdmissionPolicy()),
        on_usage=on_usage,
    )

    async def chat_completions(request: Request) -> Response:
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        raw_body = await request.json()
        req = ChatCompletionRequest.model_validate(raw_body)

        try:
            pool_spec = deps.catalog.resolve_alias(req.model)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

        est_prompt = _estimate_prompt_tokens(req.messages)
        est_total = est_prompt + req.max_tokens
        try:
            permit = deps.budget.authorize_and_reserve(
                _extract_api_key(request),
                pool_spec.name,
                est_total,
            )
        except AuthenticationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except BudgetExceededError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )

        try:
            await deps.admission.acquire(permit.priority)
        except _AdmissionShedError as exc:
            deps.budget.reconcile(permit, 0)
            return JSONResponse(
                {"error": str(exc)},
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )

        target_pool, replica = _select_pool_and_replica(
            deps=deps,
            initial_pool=pool_spec,
            messages=req.messages,
        )
        if replica is None:
            deps.admission.release()
            deps.budget.reconcile(permit, 0)
            return JSONResponse({"error": "No healthy replicas available"}, status_code=503)

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

    return Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])


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
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        completed = False
        emitted_tokens = 0
        try:
            async for chunk in deps.engine.stream(
                replica=replica,
                model_id=target_pool.model_id,
                request_id=trace_id,
                payload=req.model_dump(),
            ):
                if await request.is_disconnected():
                    break
                emitted_tokens = chunk.completion_tokens_so_far
                payload = {
                    "id": f"chatcmpl-{trace_id}",
                    "object": "chat.completion.chunk",
                    "model": target_pool.model_id,
                    "choices": [{"index": 0, "delta": {"content": chunk.delta}}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                completed = True
                yield "data: [DONE]\n\n"
        finally:
            try:
                if not completed:
                    try:
                        await deps.engine.abort(replica=replica, request_id=trace_id)
                    except RuntimeError:
                        asyncio.get_running_loop().create_task(
                            deps.engine.abort(replica=replica, request_id=trace_id)
                        )
            finally:
                total_tokens = est_prompt + emitted_tokens
                _record_usage_and_reconcile(
                    deps=deps,
                    permit=permit,
                    pool_name=target_pool.name,
                    total_tokens=total_tokens,
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


def _pick_replica(
    pool: ModelPoolSpec, replicas: list[ReplicaEndpoint], prefix: str
) -> ReplicaEndpoint:
    ordered = sorted(replicas, key=lambda r: r.replica_id)
    digest = hashlib.sha256(prefix.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "big") % len(ordered)
    candidate = ordered[idx]
    if (
        candidate.requests_waiting <= pool.keda_waiting_threshold
        and candidate.kv_cache_usage <= pool.keda_kv_cache_threshold
    ):
        return candidate
    return min(ordered, key=lambda r: (r.requests_waiting, r.kv_cache_usage, r.replica_id))


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
