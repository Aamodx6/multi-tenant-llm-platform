from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from catalog.pool_catalog import IsolationMode, ModelPoolCatalog, ModelPoolSpec, model_slug


class DeployConfig(BaseModel):
    """Cluster-level configuration applied when rendering Kubernetes manifests."""

    namespace: str = "llm"
    prometheus_url: str = "http://prometheus-operated.monitoring:9090"
    request_control_image: str = "ghcr.io/multi-tenant-llm/request-control:latest"
    vllm_image: str = "vllm/vllm-openai:v0.9.0"
    s5cmd_image: str = "peakcom/s5cmd:v2.3.0"
    model_bucket_name: str = "llm-model-weights"
    prefetch_service_account: str = "weight-prefetcher"
    use_prefetched_weights: bool = False
    cache_salt_secret_name: str = "request-control-secrets"
    cache_salt_secret_key: str = "cache-salt-secret"
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
    keda_map: dict[str, dict[str, Any]] = targets["keda_triggers"]
    slo_map: dict[str, dict[str, Any]] = targets["rollout_slos"]
    prefetch_map: dict[str, dict[str, Any]] = targets["prefetch_targets"]

    out: dict[str, str] = {}
    for name, spec in catalog.pools.items():
        args = list(vllm_args_map.get(name, []))
        keda_cfg = keda_map[name]
        out[f"deployment-{name}.yaml"] = yaml.safe_dump(
            _vllm_deployment(name, spec, args, cfg), sort_keys=False
        )
        out[f"keda-{name}.yaml"] = yaml.safe_dump(
            _keda_scaled_object(name, spec.model_id, keda_cfg, cfg), sort_keys=False
        )
        if spec.isolation == IsolationMode.SHARED:
            out[f"pdb-{name}.yaml"] = yaml.safe_dump(_pool_pdb(name, cfg), sort_keys=False)

    out["daemonset-weight-prefetch.yaml"] = yaml.safe_dump(
        _weight_prefetch_daemonset(prefetch_map, cfg), sort_keys=False
    )

    out["networkpolicy-inference.yaml"] = yaml.safe_dump(
        _inference_network_policy(cfg), sort_keys=False
    )
    out["karpenter-gpu.yaml"] = yaml.safe_dump_all(
        [_karpenter_nodepool(cfg), _karpenter_nodeclass(cfg)], sort_keys=False
    )
    ttft_slo_seconds = min(float(v["p95_ttft_ms"]) for v in slo_map.values()) / 1000.0
    out["rollout-request-control.yaml"] = yaml.safe_dump_all(
        [
            _request_control_service("request-control-stable", cfg),
            _request_control_service("request-control-canary", cfg),
            _request_control_rollout(cfg),
            _ttft_analysis_template(ttft_slo_seconds, cfg),
        ],
        sort_keys=False,
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
    effective_args = list(vllm_args)
    slug = model_slug(spec.model_id)
    init_containers: list[dict[str, Any]] = []
    if cfg.use_prefetched_weights:
        if "--model" in effective_args:
            idx = effective_args.index("--model")
            if idx + 1 < len(effective_args):
                effective_args[idx + 1] = f"/models/{slug}"
        ready_checks = [f"/models/{slug}/.ready"]
        ready_checks.extend(
            f"/models/loras/{name}/{lora.name}/.ready"
            for lora in spec.loras
            if not lora.dynamic
        )
        cond = " && ".join(f"[ -f {path} ]" for path in ready_checks)
        init_containers.append(
            {
                "name": "wait-model-weights",
                "image": "busybox:1.36",
                "command": ["/bin/sh", "-c", f"until {cond}; do sleep 2; done"],
                "volumeMounts": [{"name": "model-cache", "mountPath": "/models"}],
            }
        )

    env_vars: list[dict[str, str]] = []
    if spec.enable_lora and any(lora.dynamic for lora in spec.loras):
        env_vars.append({"name": "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "value": "true"})

    container: dict[str, Any] = {
        "name": "vllm",
        "image": cfg.vllm_image,
        "args": effective_args,
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
            "preStop": {"exec": {"command": ["/bin/sh", "-c", "sleep 15"]}}
        },
    }
    if env_vars:
        container["env"] = env_vars

    pod_spec: dict[str, Any] = {
        "terminationGracePeriodSeconds": 180,
        "nodeSelector": {"nvidia.com/gpu.present": "true"},
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ],
        "containers": [container],
        "volumes": [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
            {
                "name": "model-cache",
                "hostPath": {"path": "/mnt/nvme/models", "type": "DirectoryOrCreate"},
            },
        ],
    }
    if init_containers:
        pod_spec["initContainers"] = init_containers

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
                "spec": pod_spec,
            },
        },
    }


def _keda_scaled_object(
    name: str,
    model_id: str,
    keda_cfg: dict[str, Any],
    cfg: DeployConfig,
) -> dict[str, Any]:
    waiting_meta: dict[str, str] = {
        "serverAddress": cfg.prometheus_url,
        "metricName": "vllm_requests_waiting",
        "query": f'sum(vllm:num_requests_waiting{{model_name="{model_id}"}})',
        "threshold": str(keda_cfg["waiting_threshold"]),
    }
    if int(keda_cfg["min_replicas"]) == 0:
        waiting_meta["activationThreshold"] = "1"

    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": f"keda-{name}", "namespace": cfg.namespace},
        "spec": {
            "scaleTargetRef": {"name": f"vllm-{name}"},
            "pollingInterval": 15,
            "cooldownPeriod": 300,
            "minReplicaCount": int(keda_cfg["min_replicas"]),
            "maxReplicaCount": int(keda_cfg["max_replicas"]),
            "triggers": [
                {"type": "prometheus", "metadata": waiting_meta},
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": cfg.prometheus_url,
                        "metricName": "vllm_kv_cache_usage",
                        "query": f'avg(vllm:kv_cache_usage_perc{{model_name="{model_id}"}})',
                        "threshold": f"{float(keda_cfg['kv_cache_threshold']):.2f}",
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
                {"ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}]},
                {"to": [{"ipBlock": {"cidr": "10.0.0.0/8"}}]},
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
            "instanceStorePolicy": "RAID0",
            "userData": (
                "MIME-Version: 1.0\n"
                'Content-Type: multipart/mixed; boundary="//"\n\n'
                "--//\n"
                "Content-Type: text/x-shellscript; charset=\"us-ascii\"\n\n"
                "#!/bin/sh\n"
                "if [ -d /mnt/k8s-disks/0 ] && [ ! -e /mnt/nvme ]; then "
                "ln -sfn /mnt/k8s-disks/0 /mnt/nvme; fi\n"
                "--//--\n"
            ),
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


def _normalize_s3_uri(uri: str, bucket_name: str) -> str:
    trimmed = uri.rstrip("/")
    if trimmed.startswith("s3://models/"):
        return f"s3://{bucket_name}/{trimmed[len('s3://models/'):]}"
    return trimmed


def _weight_prefetch_daemonset(
    prefetch_targets: dict[str, dict[str, Any]],
    cfg: DeployConfig,
) -> dict[str, Any]:
    sync_entries: list[tuple[str, str]] = []
    for pool_name, entry in prefetch_targets.items():
        slug = str(entry["slug"])
        uri = _normalize_s3_uri(str(entry["prefetch_uri"]), cfg.model_bucket_name)
        sync_entries.append((slug, uri))
        for lora in entry.get("loras", []):
            sync_entries.append(
                (
                    f"loras/{pool_name}/{lora['name']}",
                    _normalize_s3_uri(str(lora["s3_uri"]), cfg.model_bucket_name),
                )
            )

    sync_commands = " && ".join(
        (
            f"(mkdir -p /models/.locks /models/.staging && "
            f"lock_name=$(printf '%s' '{rel}' | tr '/' '_') && "
            f'flock "/models/.locks/${{lock_name}}.lock" /bin/sh -c \''
            f"if [ ! -f /models/{rel}/.ready ]; then "
            f'stage="/models/.staging/${{lock_name}}.$$"; '
            f'trap "rm -rf \\"$stage\\"" EXIT; '
            f'mkdir -p "$stage" && '
            f'/tools/s5cmd cp --concurrency 32 "{uri}/*" "$stage/" && '
            f'(ls "$stage"/*.safetensors >/dev/null 2>&1 || '
            f'ls "$stage"/adapter_model* >/dev/null 2>&1) && '
            f'touch "$stage/.ready" && '
            f'mkdir -p "$(dirname /models/{rel})" && '
            f'rm -rf "/models/{rel}.old" && '
            f'([ ! -e "/models/{rel}" ] || mv "/models/{rel}" "/models/{rel}.old") && '
            f'mv "$stage" "/models/{rel}" && '
            f'rm -rf "/models/{rel}.old" && '
            f"trap - EXIT; fi')"
        )
        for rel, uri in sync_entries
    )
    script = f"set -eu; {sync_commands or 'true'}; exec sleep infinity"
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "name": "weight-prefetch",
            "namespace": cfg.namespace,
            "labels": {"app": "weight-prefetch"},
        },
        "spec": {
            "selector": {"matchLabels": {"app": "weight-prefetch"}},
            "template": {
                "metadata": {"labels": {"app": "weight-prefetch"}},
                "spec": {
                    "serviceAccountName": cfg.prefetch_service_account,
                    "nodeSelector": {"nvidia.com/gpu.present": "true"},
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ],
                    "initContainers": [
                        {
                            "name": "copy-s5cmd",
                            "image": cfg.s5cmd_image,
                            "command": ["cp", "/s5cmd", "/tools/s5cmd"],
                            "volumeMounts": [{"name": "tools", "mountPath": "/tools"}],
                        }
                    ],
                    "containers": [
                        {
                            "name": "s5cmd-prefetch",
                            "image": "alpine:3.20",
                            "command": ["/bin/sh", "-c", script],
                            "volumeMounts": [
                                {"name": "model-cache", "mountPath": "/models"},
                                {"name": "tools", "mountPath": "/tools"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "model-cache",
                            "hostPath": {"path": "/mnt/nvme/models", "type": "DirectoryOrCreate"},
                        },
                        {"name": "tools", "emptyDir": {}},
                    ],
                },
            },
        },
    }


def _request_control_service(service_name: str, cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name, "namespace": cfg.namespace},
        "spec": {
            "selector": {"app": "request-control"},
            "ports": [{"name": "http", "port": 8080, "targetPort": 8080}],
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
                    "canaryService": "request-control-canary",
                    "stableService": "request-control-stable",
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
                    ],
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
                            "env": [
                                {
                                    "name": "CACHE_SALT_SECRET",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": cfg.cache_salt_secret_name,
                                            "key": cfg.cache_salt_secret_key,
                                        }
                                    },
                                }
                            ],
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


def _ttft_analysis_template(ttft_slo_seconds: float, cfg: DeployConfig) -> dict[str, Any]:
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "AnalysisTemplate",
        "metadata": {"name": "request-control-ttft-slo", "namespace": cfg.namespace},
        "spec": {
            "metrics": [
                {
                    "name": "p95-ttft-slo",
                    "interval": "30s",
                    "count": 4,
                    "successCondition": f"result[0] <= {ttft_slo_seconds:.3f}",
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
