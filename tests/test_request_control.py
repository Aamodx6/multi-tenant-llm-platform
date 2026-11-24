import asyncio
from collections.abc import MutableMapping
from typing import Any

import httpx
import pytest

from catalog.pool_catalog import load_and_validate
from gateway.pool_health import PoolHealthStore
from gateway.request_control import (
    AdmissionPolicy,
    FakeEngineAdapter,
    ReplicaEndpoint,
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
        # Full stream used 2+3=5 tokens, aborted stream used 1+1=2 tokens -> 93 left
        tail = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-acme"},
            json={
                "model": "qwen-7b",
                "max_tokens": 90,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert tail.status_code == 200
