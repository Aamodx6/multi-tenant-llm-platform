from pathlib import Path

import httpx
import pytest
import yaml

from catalog.pool_catalog import load_and_validate
from deploy.manifest_renderer import DeployConfig, render_deploy_manifests
from gateway.server import create_default_app

CATALOG_YAML = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct-AWQ
    quantization: awq
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
"""


def test_render_deploy_manifests_wires_keda_deployments_karpenter_rollouts_and_netpol() -> None:
    catalog = load_and_validate(CATALOG_YAML)
    manifests = render_deploy_manifests(catalog, DeployConfig(namespace="llm-prod"))

    expected_keys = {
        "keda-pool-8b.yaml",
        "keda-pool-fallback.yaml",
        "deployment-pool-8b.yaml",
        "deployment-pool-fallback.yaml",
        "pdb-pool-8b.yaml",
        "networkpolicy-inference.yaml",
        "karpenter-gpu.yaml",
        "rollout-request-control.yaml",
    }
    assert expected_keys <= set(manifests)

    keda_8b = yaml.safe_load(manifests["keda-pool-8b.yaml"])
    assert keda_8b["kind"] == "ScaledObject"
    assert keda_8b["spec"]["minReplicaCount"] == 1
    assert keda_8b["spec"]["maxReplicaCount"] == 4
    triggers = keda_8b["spec"]["triggers"]
    assert any("vllm:num_requests_waiting" in t["metadata"]["query"] for t in triggers)
    assert any("vllm:kv_cache_usage_perc" in t["metadata"]["query"] for t in triggers)

    keda_fb = yaml.safe_load(manifests["keda-pool-fallback.yaml"])
    assert keda_fb["spec"]["minReplicaCount"] == 0
    assert keda_fb["spec"]["triggers"][0]["metadata"]["activationThreshold"] == "1"

    dep_8b = yaml.safe_load(manifests["deployment-pool-8b.yaml"])
    container = dep_8b["spec"]["template"]["spec"]["containers"][0]
    assert "--quantization" in container["args"]
    assert "awq" in container["args"]
    assert any(m["mountPath"] == "/dev/shm" for m in container["volumeMounts"])
    assert container["startupProbe"]["httpGet"]["path"] == "/health"

    rollout_docs = list(yaml.safe_load_all(manifests["rollout-request-control.yaml"]))
    assert rollout_docs[0]["kind"] == "Rollout"
    assert rollout_docs[1]["kind"] == "AnalysisTemplate"
    sec = rollout_docs[0]["spec"]["template"]["spec"]["containers"][0]["securityContext"]
    assert sec["runAsNonRoot"] is True
    assert sec["allowPrivilegeEscalation"] is False

    karpenter_docs = list(yaml.safe_load_all(manifests["karpenter-gpu.yaml"]))
    assert karpenter_docs[0]["kind"] == "NodePool"
    assert karpenter_docs[1]["kind"] == "EC2NodeClass"


@pytest.mark.anyio
async def test_default_catalog_and_server_app_expose_metrics() -> None:
    catalog = load_and_validate(Path("catalog/pools.yaml").read_text(encoding="utf-8"))
    manifests = render_deploy_manifests(catalog)
    assert "keda-pool-8b-awq.yaml" in manifests

    app = create_default_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "gateway_requests_total" in resp.text
