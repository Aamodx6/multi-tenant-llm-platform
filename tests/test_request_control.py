import asyncio
import json
from collections.abc import AsyncIterator, MutableMapping
from typing import Any

import httpx
import pytest

from catalog.pool_catalog import load_and_validate
from gateway.pool_health import PoolHealthStore
from gateway.request_control import (
    AdmissionPolicy,
    FakeEngineAdapter,
    ReplicaEndpoint,
    VLLMAdapter,
    create_app,
)
from gateway.tenant_budget import (
    InMemoryBudgetAdapter,
    PriorityClass,
    TenantBudget,
    TenantRecord,
    hash_api_key,
)

CATALOG_YAML = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    isolation: shared
    fallback_pool: pool-fallback
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases: [qwen-7b]
  pool-fallback:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    isolation: fallback
    min_replicas: 0
    max_replicas: 2
    keda_waiting_threshold: 6
    keda_kv_cache_threshold: 0.85
    slo_p95_ttft_ms: 600
    slo_availability: 0.99
    aliases: [qwen-1.5b]
"""


@pytest.mark.anyio
async def test_chat_completions_authenticates_reserves_dispatches_and_reconciles() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=100)
    adapter = InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant})
    budget = TenantBudget(adapter=adapter)
    health = PoolHealthStore()
    engine = FakeEngineAdapter(
        replicas={
            "pool-8b": [ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000")],
            "pool-fallback": [ReplicaEndpoint(replica_id="fb-0", url="http://fb-0:8000")],
        },
        completion_tokens=15,
    )
    app = create_app(catalog=catalog, budget=budget, pool_health=health, engine=engine)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauth = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauth.status_code == 401

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 70,
                "messages": [{"role": "user", "content": "hello world"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert resp.headers["x-served-pool"] == "pool-8b"
        assert resp.headers["x-trace-id"]

        # Prompt ~2 tokens + max_tokens 70 = 72 reserved, actual 17 -> 83 tokens left -> 60 succeeds
        second = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 55,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert second.status_code == 200

        # Req 2 used 16 actual -> 67 tokens left -> requesting 71 exceeds remaining TPM -> 429
        over = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 70,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert over.status_code == 429
        assert int(over.headers["retry-after"]) >= 1


@pytest.mark.anyio
async def test_prefix_affinity_bounded_load_and_pool_health_fallback() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=10_000)
    budget = TenantBudget(adapter=InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant}))
    health = PoolHealthStore()
    replicas_8b = [
        ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000"),
        ReplicaEndpoint(replica_id="8b-1", url="http://8b-1:8000"),
        ReplicaEndpoint(replica_id="8b-2", url="http://8b-2:8000"),
    ]
    engine = FakeEngineAdapter(
        replicas={
            "pool-8b": replicas_8b,
            "pool-fallback": [ReplicaEndpoint(replica_id="fb-0", url="http://fb-0:8000")],
        }
    )
    app = create_app(catalog=catalog, budget=budget, pool_health=health, engine=engine)

    shared_sys = {"role": "system", "content": "You are a strict Python reviewer."}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={"model": "qwen-7b", "messages": [shared_sys, {"role": "user", "content": "a"}]},
        )
        r2 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={"model": "qwen-7b", "messages": [shared_sys, {"role": "user", "content": "b"}]},
        )
        assert r1.status_code == 200 and r2.status_code == 200
        preferred_id = r1.headers["x-served-replica"]
        assert r2.headers["x-served-replica"] == preferred_id

        # Saturate the preferred replica -> bounded load routes to an idle replica
        for rep in replicas_8b:
            if rep.replica_id == preferred_id:
                rep.requests_waiting = 10
                rep.kv_cache_usage = 0.92
        r3 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={"model": "qwen-7b", "messages": [shared_sys, {"role": "user", "content": "c"}]},
        )
        assert r3.status_code == 200
        assert r3.headers["x-served-replica"] != preferred_id

        # Mark pool-8b degraded -> falls back to pool-fallback
        health.set_degraded("pool-8b", reason="kv_cache_high", ttl_seconds=60.0)
        r4 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={"model": "qwen-7b", "messages": [shared_sys, {"role": "user", "content": "d"}]},
        )
        assert r4.status_code == 200
        assert r4.headers["x-served-pool"] == "pool-fallback"
        assert r4.json()["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.anyio
async def test_local_priority_admission_sheds_free_before_paid_and_refunds_permit() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    paid_tenant = TenantRecord(tenant_id="paid-co", priority=PriorityClass.PAID, tpm_limit=10_000)
    free_tenant = TenantRecord(tenant_id="free-co", priority=PriorityClass.FREE, tpm_limit=100)
    budget = TenantBudget(
        adapter=InMemoryBudgetAdapter(
            {
                hash_api_key("sk-paid"): paid_tenant,
                hash_api_key("sk-free"): free_tenant,
            }
        )
    )
    gate = asyncio.Event()
    in_flight_started = asyncio.Event()
    engine = FakeEngineAdapter(
        replicas={"pool-8b": [ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000")]},
        completion_tokens=10,
        complete_gate=gate,
        on_start=in_flight_started,
    )
    app = create_app(
        catalog=catalog,
        budget=budget,
        pool_health=PoolHealthStore(),
        engine=engine,
        admission_policy=AdmissionPolicy(
            max_in_flight=1,
            estimated_request_ms=300,
            free_wait_budget_ms=100,
            paid_wait_budget_ms=2000,
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-paid"},
                json={"model": "qwen-7b", "messages": [{"role": "user", "content": "hold"}]},
            )
        )
        await in_flight_started.wait()

        # Free request expects 300ms wait > 100ms budget -> shed immediately with 429
        free_resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-free"},
            json={
                "model": "qwen-7b",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert free_resp.status_code == 429
        assert int(free_resp.headers["retry-after"]) >= 1

        # Paid request expects 300ms wait <= 2000ms budget -> queued until first finishes
        paid_queued = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-paid"},
                json={"model": "qwen-7b", "messages": [{"role": "user", "content": "queued"}]},
            )
        )
        await asyncio.sleep(0.02)
        gate.set()

        r1, r2 = await asyncio.gather(first_task, paid_queued)
        assert r1.status_code == 200
        assert r2.status_code == 200

        # Free tenant's shed permit was reconciled to 0 -> full 100 TPM still intact
        free_ok = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-free"},
            json={
                "model": "qwen-7b",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert free_ok.status_code == 200


@pytest.mark.anyio
async def test_sse_streaming_passthrough_and_client_disconnect_aborts_engine() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=100)
    adapter = InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant})
    budget = TenantBudget(adapter=adapter)
    engine = FakeEngineAdapter(
        replicas={"pool-8b": [ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000")]},
        stream_chunks=["tok1", "tok2", "tok3"],
    )
    app = create_app(
        catalog=catalog,
        budget=budget,
        pool_health=PoolHealthStore(),
        engine=engine,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        lines: list[str] = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme", "x-trace-id": "trace-full"},
            json={
                "model": "qwen-7b",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "one two"}],
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)

        assert lines[-1] == "data: [DONE]"
        assert len(lines) == 4  # 3 chunks + [DONE]
        assert engine.aborted == []

        # Early ASGI http.disconnect after first chunk -> abort called, partial usage reconciled
        chunk_gate = asyncio.Event()
        first_chunk_sent = asyncio.Event()
        engine.stream_gate = chunk_gate
        sent_events: list[MutableMapping[str, Any]] = []
        sent_chunks = 0

        async def receive() -> MutableMapping[str, Any]:
            nonlocal sent_chunks
            if sent_chunks == 0:
                sent_chunks += 1
                body = (
                    b'{"model":"qwen-7b","max_tokens":50,"stream":true,'
                    b'"messages":[{"role":"user","content":"one"}]}'
                )
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await first_chunk_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            sent_events.append(message)
            if message.get("type") == "http.response.body" and message.get("body"):
                first_chunk_sent.set()

        scope: MutableMapping[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"authorization", b"Bearer sk-acme"),
                (b"x-trace-id", b"trace-abort"),
                (b"content-type", b"application/json"),
            ],
        }
        await app(scope, receive, send)
        engine.stream_gate = None

        assert engine.aborted == [("8b-0", "trace-abort")]
        # Full stream used 7+3=10 tokens, aborted stream used 5+1=6 tokens -> 84 left
        tail = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 75,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert tail.status_code == 200


@pytest.mark.anyio
async def test_openai_error_envelope_400_validation_and_sse_fields() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=1000)
    budget = TenantBudget(adapter=InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant}))
    engine = FakeEngineAdapter(
        replicas={"pool-8b": [ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000")]},
        stream_chunks=["a", "b"],
    )
    app = create_app(catalog=catalog, budget=budget, pool_health=PoolHealthStore(), engine=engine)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        bad_json = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme", "Content-Type": "application/json"},
            content=b"{not-json",
        )
        assert bad_json.status_code == 400
        assert bad_json.json()["error"]["type"] == "invalid_request_error"

        bad_schema = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={"model": "qwen-7b"},
        )
        assert bad_schema.status_code == 400
        assert bad_schema.json()["error"]["type"] == "invalid_request_error"

        unauth = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauth.status_code == 401
        assert unauth.json()["error"]["code"] == "invalid_api_key"

        ok = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert ok.status_code == 200
        assert isinstance(ok.json()["created"], int)

        chunks: list[dict[str, Any]] = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))

        assert isinstance(chunks[0]["created"], int)
        assert chunks[0]["choices"][0]["finish_reason"] is None
        assert chunks[1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] > 0


@pytest.mark.anyio
async def test_queued_disconnect_refunds_budget_and_chwbl_minimizes_reshuffle() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(
        tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=100, max_concurrency=1
    )
    gate = asyncio.Event()
    started = asyncio.Event()
    replicas = [
        ReplicaEndpoint(replica_id="8b-0", url="http://8b-0:8000"),
        ReplicaEndpoint(replica_id="8b-1", url="http://8b-1:8000"),
        ReplicaEndpoint(replica_id="8b-2", url="http://8b-2:8000"),
    ]
    engine = FakeEngineAdapter(
        replicas={"pool-8b": replicas},
        completion_tokens=5,
        complete_gate=gate,
        on_start=started,
    )
    blocker = TenantRecord(
        tenant_id="blocker", priority=PriorityClass.PAID, tpm_limit=1000, max_concurrency=4
    )
    budget2 = TenantBudget(
        adapter=InMemoryBudgetAdapter(
            {hash_api_key("sk-acme"): tenant, hash_api_key("sk-block"): blocker}
        )
    )
    app2 = create_app(
        catalog=catalog,
        budget=budget2,
        pool_health=PoolHealthStore(),
        engine=engine,
        admission_policy=AdmissionPolicy(
            max_in_flight=1,
            estimated_request_ms=100,
            paid_wait_budget_ms=5000,
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app2), base_url="http://test"
    ) as client:
        t_hold = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-block"},
                json={"model": "qwen-7b", "messages": [{"role": "user", "content": "hold"}]},
            )
        )
        await started.wait()

        # Queued acme request cancelled mid-queue
        t_queued = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-acme"},
                json={
                    "model": "qwen-7b",
                    "max_tokens": 90,
                    "messages": [{"role": "user", "content": "wait"}],
                },
            )
        )
        await asyncio.sleep(0.02)
        t_queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t_queued

        gate.set()
        await t_hold

        # Fast-path must still work and acme's concurrency + 90 tokens must be refunded
        engine_fast = FakeEngineAdapter(replicas={"pool-8b": replicas}, completion_tokens=5)
        app3 = create_app(
            catalog=catalog,
            budget=budget2,
            pool_health=PoolHealthStore(),
            engine=engine_fast,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app3), base_url="http://test"
        ) as c3:
            after = await c3.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-acme"},
                json={
                    "model": "qwen-7b",
                    "max_tokens": 90,
                    "messages": [{"role": "user", "content": "ok"}],
                },
            )
            assert after.status_code == 200

    # CHWBL consistency check: adding a 4th replica moves <= ~35% of 20 prefixes (not ~75%)
    engine_ring = FakeEngineAdapter(replicas={"pool-8b": replicas})
    app_ring = create_app(
        catalog=catalog,
        budget=TenantBudget(
            adapter=InMemoryBudgetAdapter(
                {hash_api_key("sk-b"): TenantRecord("b", PriorityClass.PAID, 100_000)}
            )
        ),
        pool_health=PoolHealthStore(),
        engine=engine_ring,
    )
    before_map: dict[str, str] = {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_ring), base_url="http://test"
    ) as client:
        for i in range(20):
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-b"},
                json={
                    "model": "qwen-7b",
                    "messages": [{"role": "system", "content": f"sys-prompt-{i}"}],
                },
            )
            before_map[f"sys-prompt-{i}"] = r.headers["x-served-replica"]

        expanded_replicas = [*replicas, ReplicaEndpoint(replica_id="8b-3", url="http://8b-3:8000")]
        app_ring_expanded = create_app(
            catalog=catalog,
            budget=TenantBudget(
                adapter=InMemoryBudgetAdapter(
                    {hash_api_key("sk-b"): TenantRecord("b", PriorityClass.PAID, 100_000)}
                )
            ),
            pool_health=PoolHealthStore(),
            engine=FakeEngineAdapter(replicas={"pool-8b": expanded_replicas}),
        )
        moved = 0
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_ring_expanded), base_url="http://test"
        ) as client_expanded:
            for i in range(20):
                r = await client_expanded.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-b"},
                    json={
                        "model": "qwen-7b",
                        "messages": [{"role": "system", "content": f"sys-prompt-{i}"}],
                    },
                )
                if r.headers["x-served-replica"] != before_map[f"sys-prompt-{i}"]:
                    moved += 1
        assert moved <= 9


@pytest.mark.anyio
async def test_vllm_adapter_parses_sse_usage_aborts_by_closing_stream_and_scrapes_metrics() -> None:
    seen_payloads: list[dict[str, Any]] = []
    seen_urls: list[str] = []
    stream_closed = False

    class _MockSSEStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":{"prompt_tokens":4,'
                b'"completion_tokens":2,"total_tokens":6}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            nonlocal stream_closed
            stream_closed = True

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    "# HELP vllm:num_requests_waiting Waiting\n"
                    'vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"} 3.0\n'
                    'vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-7B-Instruct"} 0.42\n'
                ),
            )
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_MockSSEStream(),
        )

    rep = ReplicaEndpoint(replica_id="8b-0", url="http://vllm-0:8000")
    adapter = VLLMAdapter(
        replicas={"pool-8b": [rep]},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await adapter.scrape_metrics()
    assert rep.requests_waiting == 3
    assert rep.kv_cache_usage == pytest.approx(0.42)

    chunks = [
        c
        async for c in adapter.stream(
            replica=rep,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            request_id="req-1",
            payload={"messages": [{"role": "user", "content": "hi"}], "temperature": 0.1},
        )
    ]
    assert seen_payloads[0]["stream_options"] == {"include_usage": True}
    assert seen_payloads[0]["temperature"] == 0.1
    assert [c.delta for c in chunks if c.delta] == ["hel", "lo"]
    assert chunks[-1].prompt_tokens == 4
    assert chunks[-1].completion_tokens_so_far == 2

    # Abort must NOT call /v1/abort
    await adapter.abort(replica=rep, request_id="req-1")
    assert all("/v1/abort" not in u for u in seen_urls)
    assert stream_closed is True

    # End-to-end through RequestControl: preserves role="assistant" & terminal finish_reason="stop"
    catalog = load_and_validate(CATALOG_YAML)
    tenant = TenantRecord(tenant_id="acme", priority=PriorityClass.PAID, tpm_limit=1000)
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(adapter=InMemoryBudgetAdapter({hash_api_key("sk-acme"): tenant})),
        pool_health=PoolHealthStore(),
        engine=adapter,
    )
    gw_chunks: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp,
    ):
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                gw_chunks.append(json.loads(line[6:]))
    assert gw_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert gw_chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert gw_chunks[-1]["usage"]["total_tokens"] == 6


@pytest.mark.anyio
async def test_dynamic_lora_enforces_tenant_ownership_blocks_incompatible_fallback_and_loads() -> (
    None
):
    lora_catalog_yaml = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    fallback_pool: pool-fallback
    enable_lora: true
    max_loras: 4
    max_lora_rank: 16
    max_cpu_loras: 8
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    loras:
      - name: tenant-a-sql
        tenant_id: tenant-a
        s3_uri: s3://models/loras/tenant-a-sql
        rank: 16
        dynamic: true
  pool-fallback:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    isolation: fallback
    min_replicas: 0
    max_replicas: 2
    keda_waiting_threshold: 6
    keda_kv_cache_threshold: 0.85
    slo_p95_ttft_ms: 600
    slo_availability: 0.99
"""
    catalog = load_and_validate(lora_catalog_yaml)
    ta = TenantRecord("tenant-a", PriorityClass.PAID, tpm_limit=100, max_concurrency=1)
    tb = TenantRecord("tenant-b", PriorityClass.PAID, tpm_limit=100, max_concurrency=1)
    health = PoolHealthStore()
    engine = FakeEngineAdapter(
        replicas={
            "pool-8b": [ReplicaEndpoint("8b-0", "http://8b-0:8000")],
            "pool-fallback": [ReplicaEndpoint("fb-0", "http://fb-0:8000")],
        },
        completion_tokens=5,
    )
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(
            adapter=InMemoryBudgetAdapter(
                {hash_api_key("sk-a"): ta, hash_api_key("sk-b"): tb}
            )
        ),
        pool_health=health,
        engine=engine,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Tenant B cannot access Tenant A's private LoRA (returns 404 and refunds budget)
        denied = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-b"},
            json={
                "model": "tenant-a-sql",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert denied.status_code == 404

        # Tenant B's budget was NOT consumed by denied call
        b_base = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-b"},
            json={
                "model": "pool-8b",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert b_base.status_code == 200

        # Tenant A invokes its LoRA cleanly
        allowed = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-a"},
            json={
                "model": "tenant-a-sql",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "select *"}],
            },
        )
        assert allowed.status_code == 200
        assert allowed.json()["model"] == "tenant-a-sql"
        assert engine.dispatched[-1][1] == "tenant-a-sql"

        # When pool-8b is degraded, LoRA must NOT fall back to 1.5B pool-fallback -> 503
        health.set_degraded("pool-8b", "kv_pressure", ttl_seconds=60.0)
        no_fb = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-a"},
            json={
                "model": "tenant-a-sql",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "select *"}],
            },
        )
        assert no_fb.status_code == 503

    # VLLMAdapter loads dynamic LoRA via /v1/load_lora_adapter once per (replica, lora)
    load_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/load_lora_adapter":
            load_calls.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"status": "loaded"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    rep = ReplicaEndpoint("8b-0", "http://vllm-0:8000")
    vllm = VLLMAdapter(
        replicas={"pool-8b": [rep]},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await vllm.ensure_lora_loaded(
        replica=rep,
        lora_name="tenant-a-sql",
        lora_path="/models/loras/pool-8b/tenant-a-sql",
    )
    await vllm.ensure_lora_loaded(
        replica=rep,
        lora_name="tenant-a-sql",
        lora_path="/models/loras/pool-8b/tenant-a-sql",
    )
    assert load_calls == [
        {
            "lora_name": "tenant-a-sql",
            "lora_path": "/models/loras/pool-8b/tenant-a-sql",
        }
    ]


@pytest.mark.anyio
async def test_prefix_cache_salt_isolates_tenants_and_strips_spoofed_salt() -> None:
    catalog = load_and_validate(
        """
pools:
  pool-shared:
    model_id: Qwen/Qwen2.5-7B-Instruct
    isolation: shared
    isolate_prefix_cache: true
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
  pool-optout:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    isolation: dedicated
    isolate_prefix_cache: false
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    )
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    vllm = VLLMAdapter(
        replicas={
            "pool-shared": [ReplicaEndpoint("shared-0", "http://shared-0:8000")],
            "pool-optout": [ReplicaEndpoint("opt-0", "http://opt-0:8000")],
        },
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app(
        catalog=catalog,
        budget=TenantBudget(
            adapter=InMemoryBudgetAdapter(
                {
                    hash_api_key("sk-a"): TenantRecord("tenant-a", PriorityClass.PAID, 10_000),
                    hash_api_key("sk-b"): TenantRecord("tenant-b", PriorityClass.PAID, 10_000),
                }
            )
        ),
        pool_health=PoolHealthStore(),
        engine=vllm,
        cache_salt_secret=b"unit-test-secret",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        r_a = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-a"},
            json={
                "model": "pool-shared",
                "cache_salt": "attacker-forged-salt",
                "messages": [{"role": "system", "content": "shared system prefix"}],
            },
        )
        r_b = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-b"},
            json={
                "model": "pool-shared",
                "cache_salt": "attacker-forged-salt",
                "messages": [{"role": "system", "content": "shared system prefix"}],
            },
        )
        r_opt = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-a"},
            json={
                "model": "pool-optout",
                "cache_salt": "attacker-forged-salt",
                "messages": [{"role": "system", "content": "shared system prefix"}],
            },
        )
        assert r_a.status_code == 200 and r_b.status_code == 200 and r_opt.status_code == 200

    assert len(seen_payloads) == 3
    salt_a = seen_payloads[0]["cache_salt"]
    salt_b = seen_payloads[1]["cache_salt"]
    assert isinstance(salt_a, str) and len(salt_a) == 64
    assert salt_a != "attacker-forged-salt"
    assert salt_b != "attacker-forged-salt"
    assert salt_a != salt_b
    assert "cache_salt" not in seen_payloads[2]


@pytest.mark.anyio
async def test_vllm_adapter_singleflight_and_reloads_lora_on_404() -> None:
    load_calls: list[dict[str, Any]] = []
    chat_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url.path == "/v1/load_lora_adapter":
            load_calls.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"status": "loaded"})
        chat_attempts += 1
        if chat_attempts == 1:
            return httpx.Response(404, json={"error": {"message": "adapter evicted"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "reloaded-ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    rep = ReplicaEndpoint("8b-0", "http://vllm-0:8000")
    vllm = VLLMAdapter(
        replicas={"pool-8b": [rep]},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await asyncio.gather(
        vllm.ensure_lora_loaded(
            replica=rep, lora_name="tenant-a-sql", lora_path="/models/loras/pool-8b/tenant-a-sql"
        ),
        vllm.ensure_lora_loaded(
            replica=rep, lora_name="tenant-a-sql", lora_path="/models/loras/pool-8b/tenant-a-sql"
        ),
    )
    assert len(load_calls) == 1

    res = await vllm.complete(
        replica=rep,
        model_id="tenant-a-sql",
        request_id="req-1",
        payload={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.content == "reloaded-ok"
    assert len(load_calls) == 2
    assert chat_attempts == 2
