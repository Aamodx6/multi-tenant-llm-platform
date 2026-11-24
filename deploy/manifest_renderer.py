from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from catalog.pool_catalog import IsolationMode, ModelPoolCatalog, ModelPoolSpec


class DeployConfig(BaseModel):
    """Cluster-level configuration applied when rendering Kubernetes manifests."""

    namespace: str = "llm"
    prometheus_url: str = "http://prometheus-operated.monitoring:9090"
    request_control_image: str = "ghcr.io/multi-tenant-llm/request-control:latest"
    vllm_image: str = "vllm/vllm-openai:v0.7.3"
    cluster_name: str = "llm-prod-eks"
    spot_instance_families: list[str] = Field(default_factory=lambda: ["g6", "g5"])


def render_deploy_manifests(
    catalog: ModelPoolCatalog,
    config: DeployConfig | None = None,
) -> dict[str, str]:
    """Render KEDA, vLLM Deployment, PDB, NetworkPolicy, Karpenter, and Argo Rollouts YAML."""
    cfg = config or DeployConfig()
    targets = catalog.render_targets()
    vllm_args_map: dict[str, list[str]] = targets["vllm_args"]

    out: dict[str, str] = {}
    for name, spec in catalog.pools.items():
        args = vllm_args_map.get(name, [])
        out[f"deployment-{name}.yaml"] = yaml.safe_dump(
            _vllm_deployment(name, spec, args, cfg), sort_keys=False
        )
        out[f"keda-{name}.yaml"] = yaml.safe_dump(
            _keda_scaled_object(name, spec, cfg), sort_keys=False
        )
        if spec.isolation == IsolationMode.SHARED:
            out[f"pdb-{name}.yaml"] = yaml.safe_dump(_pool_pdb(name, cfg), sort_keys=False)

    out["networkpolicy-inference.yaml"] = yaml.safe_dump(
        _inference_network_policy(cfg), sort_keys=False
    )
    out["karpenter-gpu.yaml"] = yaml.safe_dump_all(
        [_karpenter_nodepool(cfg), _karpenter_nodeclass(cfg)], sort_keys=False
    )
    out["rollout-request-control.yaml"] = yaml.safe_dump_all(
        [_request_control_rollout(cfg), _ttft_analysis_template(cfg)], sort_keys=False
    )
    return out


def write_manifests(manifests: dict[str, str], out_dir: Path) -> None:
    """Write rendered manifest dictionary to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in manifests.items():
        (out_dir / filename).write_text(content, encoding="utf-8")


def _vllm_deployment(
    name: str,
    spec: ModelPoolSpec,
    vllm_args: list[str],
    cfg: DeployConfig,
) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"vllm-{name}",
            "namespace": cfg.namespace,
            "labels": {"app": "vllm", "pool": name},
        },
        "spec": {
            "replicas": max(1, spec.min_replicas),
            "selector": {"matchLabels": {"app": "vllm", "pool": name}},
            "template": {
                "metadata": {"labels": {"app": "vllm", "pool": name}},
                "spec": {
                    "terminationGracePeriodSeconds": 180,
                    "nodeSelector": {"nvidia.com/gpu.present": "true"},
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "vllm",
                            "image": cfg.vllm_image,
                            "args": vllm_args,
                            "ports": [{"containerPort": 8000, "name": "http"}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": 1},
                                "requests": {"nvidia.com/gpu": 1, "memory": "16Gi", "cpu": "4"},
                            },
                            "volumeMounts": [
                                {"name": "dshm", "mountPath": "/dev/shm"},
                                {"name": "model-cache", "mountPath": "/models"},
                            ],
                            "startupProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "periodSeconds": 5,
                                "failureThreshold": 60,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/v1/models", "port": 8000},
                                "periodSeconds": 10,
                                "timeoutSeconds": 3,
                            },
                            "lifecycle": {
                                "preStop": {
                                    "exec": {"command": ["/bin/sh", "-c", "sleep 15"]}
                                }
                            },
                        }
                    ],
                    "volumes": [
                        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
                        {
                            "name": "model-cache",
                            "hostPath": {"path": "/mnt/nvme/models", "type": "DirectoryOrCreate"},
                        },
                    ],
                },
            },
        },
    }


def _keda_scaled_object(
    name: str,
    spec: ModelPoolSpec,
    cfg: DeployConfig,
) -> dict[str, Any]:
    waiting_meta: dict[str, str] = {
        "serverAddress": cfg.prometheus_url,
        "metricName": "vllm_requests_waiting",
        "query": f'sum(vllm:num_requests_waiting{{model_name="{spec.model_id}"}})',
        "threshold": str(spec.keda_waiting_threshold),
    }
    if spec.min_replicas == 0:
        waiting_meta["activationThreshold"] = "1"

    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": f"keda-{name}", "namespace": cfg.namespace},
        "spec": {
            "scaleTargetRef": {"name": f"vllm-{name}"},
            "pollingInterval": 15,
            "cooldownPeriod": 300,
            "minReplicaCount": spec.min_replicas,
            "maxReplicaCount": spec.max_replicas,
            "triggers": [
                {"type": "prometheus", "metadata": waiting_meta},
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": cfg.prometheus_url,
                        "metricName": "vllm_kv_cache_usage",
                        "query": f'avg(vllm:kv_cache_usage_perc{{model_name="{spec.model_id}"}})',
                        "threshold": f"{spec.keda_kv_cache_threshold:.2f}",
                    },
                },
            ],
        },
    }


def _pool_pdb(name: str, cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {"name": f"pdb-{name}", "namespace": cfg.namespace},
        "spec": {
            "minAvailable": 1,
            "selector": {"matchLabels": {"app": "vllm", "pool": name}},
        },
    }


def _inference_network_policy(cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "inference-isolation", "namespace": cfg.namespace},
        "spec": {
            "podSelector": {"matchLabels": {"app": "vllm"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [{"podSelector": {"matchLabels": {"app": "request-control"}}}],
                    "ports": [{"protocol": "TCP", "port": 8000}],
                },
                {
                    "from": [{"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}],
                    "ports": [{"protocol": "TCP", "port": 8000}],
                },
            ],
            "egress": [
                {"ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 443}]}
            ],
        },
    }


def _karpenter_nodepool(cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "karpenter.sh/v1",
        "kind": "NodePool",
        "metadata": {"name": "gpu-inference"},
        "spec": {
            "template": {
                "metadata": {"labels": {"nvidia.com/gpu.present": "true"}},
                "spec": {
                    "taints": [
                        {"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}
                    ],
                    "requirements": [
                        {
                            "key": "karpenter.sh/capacity-type",
                            "operator": "In",
                            "values": ["spot", "on-demand"],
                        },
                        {
                            "key": "karpenter.k8s.aws/instance-family",
                            "operator": "In",
                            "values": list(cfg.spot_instance_families),
                        },
                    ],
                    "nodeClassRef": {
                        "group": "karpenter.k8s.aws",
                        "kind": "EC2NodeClass",
                        "name": "gpu-nvme",
                    },
                },
            },
            "limits": {"nvidia.com/gpu": "16"},
            "disruption": {
                "consolidationPolicy": "WhenEmptyOrUnderutilized",
                "consolidateAfter": "5m",
            },
        },
    }


def _karpenter_nodeclass(cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "karpenter.k8s.aws/v1",
        "kind": "EC2NodeClass",
        "metadata": {"name": "gpu-nvme"},
        "spec": {
            "amiSelectorTerms": [{"alias": "al2023@latest"}],
            "role": f"{cfg.cluster_name}-karpenter-node",
            "subnetSelectorTerms": [{"tags": {"karpenter.sh/discovery": cfg.cluster_name}}],
            "securityGroupSelectorTerms": [{"tags": {"karpenter.sh/discovery": cfg.cluster_name}}],
            "blockDeviceMappings": [
                {
                    "deviceName": "/dev/xvda",
                    "ebs": {
                        "volumeSize": "100Gi",
                        "volumeType": "gp3",
                        "deleteOnTermination": True,
                    },
                }
            ],
        },
    }


def _request_control_rollout(cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Rollout",
        "metadata": {"name": "request-control", "namespace": cfg.namespace},
        "spec": {
            "replicas": 2,
            "selector": {"matchLabels": {"app": "request-control"}},
            "strategy": {
                "canary": {
                    "steps": [
                        {"setWeight": 5},
                        {"pause": {"duration": "2m"}},
                        {"setWeight": 25},
                        {
                            "analysis": {
                                "templates": [{"templateName": "request-control-ttft-slo"}]
                            }
                        },
                        {"setWeight": 100},
                    ]
                }
            },
            "template": {
                "metadata": {"labels": {"app": "request-control"}},
                "spec": {
                    "containers": [
                        {
                            "name": "request-control",
                            "image": cfg.request_control_image,
                            "ports": [{"containerPort": 8080}],
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 10001,
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                            },
                        }
                    ]
                },
            },
        },
    }


def _ttft_analysis_template(cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "AnalysisTemplate",
        "metadata": {"name": "request-control-ttft-slo", "namespace": cfg.namespace},
        "spec": {
            "metrics": [
                {
                    "name": "p95-ttft-under-1s",
                    "interval": "30s",
                    "count": 4,
                    "successCondition": "result[0] <= 1.0",
                    "provider": {
                        "prometheus": {
                            "address": cfg.prometheus_url,
                            "query": (
                                "histogram_quantile(0.95, "
                                "sum(rate(gateway_ttft_seconds_bucket[2m])) by (le))"
                            ),
                        }
                    },
                }
            ]
        },
    }
