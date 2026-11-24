from __future__ import annotations

import json
from typing import Any

import yaml

from catalog.pool_catalog import ModelPoolCatalog

_DCGM_COUNTERS_CSV = """# NVIDIA DCGM Exporter explicit counters (pre-4.9 and canonical GPU metrics)
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %).
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (in MiB).
DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (in MiB).
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in C).
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W).
DCGM_FI_DEV_XID_ERRORS, gauge, Value of the last XID error encountered.
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, counter, Total number of single-bit volatile ECC errors.
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, counter, Total number of double-bit volatile ECC errors.
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, counter, Total number of NVLink bandwidth counters.
"""


def _burn_rate_expr(
    pool: str, short_w: str, long_w: str, multiplier: float, err_budget: float
) -> str:
    threshold = f"{multiplier * err_budget:.6f}"
    parts = []
    for win in (short_w, long_w):
        err = f'sum(rate(gateway_requests_total{{pool="{pool}",status=~"5.."}}[{win}]))'
        tot = f'clamp_min(sum(rate(gateway_requests_total{{pool="{pool}"}}[{win}])), 1e-6)'
        parts.append(f"({err} / {tot}) > {threshold}")
    return f"{parts[0]} and {parts[1]}"


def render_observability_bundle(catalog: ModelPoolCatalog) -> dict[str, str]:
    """Render Prometheus alerts, scrape config, DCGM DaemonSet/CSV, and Grafana JSON."""
    pool_rules: list[dict[str, Any]] = []
    vllm_targets: list[str] = []
    for name, spec in catalog.pools.items():
        vllm_targets.append(f"vllm-{name}:8000")
        ttft_sec = spec.slo_p95_ttft_ms / 1000.0
        err_budget = max(1e-5, 1.0 - spec.slo_availability)
        pool_rules.extend(
            [
                {
                    "alert": f"PoolHighTTFT_{name}",
                    "expr": (
                        f'histogram_quantile(0.95, sum(rate(gateway_ttft_seconds_bucket'
                        f'{{pool="{name}"}}[5m])) by (le)) > {ttft_sec:.3f}'
                    ),
                    "for": "5m",
                    "labels": {"severity": "warning", "pool": name},
                    "annotations": {
                        "summary": f"p95 TTFT breached {spec.slo_p95_ttft_ms}ms SLO on {name}",
                    },
                },
                {
                    "alert": f"PoolErrorBudgetFastBurnRate_{name}",
                    "expr": _burn_rate_expr(name, "5m", "1h", 14.4, err_budget),
                    "labels": {"severity": "critical", "pool": name},
                    "annotations": {
                        "summary": f"Fast error-budget burn rate on {name} (14.4x over 5m & 1h)",
                    },
                },
                {
                    "alert": f"PoolErrorBudgetSlowBurnRate_{name}",
                    "expr": _burn_rate_expr(name, "30m", "6h", 6.0, err_budget),
                    "labels": {"severity": "warning", "pool": name},
                    "annotations": {
                        "summary": f"Slow error-budget burn rate on {name} (6x over 30m & 6h)",
                    },
                },
                {
                    "alert": f"VLLMKVCacheHigh_{name}",
                    "expr": (
                        f'avg(vllm:kv_cache_usage_perc{{model_name="{spec.model_id}"}}) '
                        f'> {max(spec.keda_kv_cache_threshold, 0.90):.2f}'
                    ),
                    "for": "2m",
                    "labels": {"severity": "warning", "pool": name},
                    "annotations": {"summary": f"vLLM KV cache pressure elevated on {name}"},
                },
            ]
        )

    gpu_rules: list[dict[str, Any]] = [
        {
            "alert": "DCGMXidError",
            "expr": "increase(DCGM_FI_DEV_XID_ERRORS[5m]) > 0",
            "for": "0m",
            "labels": {"severity": "critical"},
            "annotations": {"summary": "GPU XID error detected on node"},
        },
    ]

    alerts_doc = {
        "groups": [
            {"name": "llm-platform-slo-burn-rate", "rules": pool_rules},
            {"name": "llm-platform-gpu-health", "rules": gpu_rules},
        ]
    }
    scrape_doc = {
        "global": {"scrape_interval": "15s"},
        "rule_files": ["alerts.yaml"],
        "scrape_configs": [
            {
                "job_name": "request-control",
                "static_configs": [{"targets": ["request-control:8080"]}],
            },
            {
                "job_name": "vllm-engines",
                "static_configs": [{"targets": vllm_targets}],
            },
            {
                "job_name": "dcgm-exporter",
                "static_configs": [{"targets": ["dcgm-exporter:9400"]}],
            },
        ],
    }
    dcgm_k8s_docs = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "dcgm-counters", "namespace": "monitoring"},
            "data": {"counters.csv": _DCGM_COUNTERS_CSV},
        },
        {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {"name": "dcgm-exporter", "namespace": "monitoring"},
            "spec": {
                "selector": {"matchLabels": {"app": "dcgm-exporter"}},
                "template": {
                    "metadata": {"labels": {"app": "dcgm-exporter"}},
                    "spec": {
                        "nodeSelector": {"nvidia.com/gpu.present": "true"},
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
                        "containers": [
                            {
                                "name": "dcgm-exporter",
                                "image": "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04",
                                "args": ["-f", "/etc/dcgm-exporter/counters.csv"],
                                "ports": [{"containerPort": 9400, "name": "metrics"}],
                                "volumeMounts": [
                                    {
                                        "name": "counters",
                                        "mountPath": "/etc/dcgm-exporter/counters.csv",
                                        "subPath": "counters.csv",
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "counters", "configMap": {"name": "dcgm-counters"}}
                        ],
                    },
                },
            },
        },
    ]
    dashboard_doc = {
        "title": "Multi-Tenant LLM Serving Platform",
        "schemaVersion": 39,
        "panels": [
            {
                "id": 1,
                "title": "p95 TTFT by Pool (vLLM & RequestControl)",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": (
                            "histogram_quantile(0.95, "
                            "sum(rate(gateway_ttft_seconds_bucket[5m])) by (le, pool))"
                        )
                    },
                    {
                        "expr": (
                            "histogram_quantile(0.95, "
                            "sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) "
                            "by (le, model_name))"
                        )
                    },
                ],
            },
            {
                "id": 2,
                "title": "vLLM Queue Depth & KV Cache Usage",
                "type": "timeseries",
                "targets": [
                    {"expr": "vllm:num_requests_waiting"},
                    {"expr": "vllm:kv_cache_usage_perc"},
                ],
            },
            {
                "id": 3,
                "title": "DCGM GPU Utilization, Temp & XID Errors",
                "type": "timeseries",
                "targets": [
                    {"expr": "DCGM_FI_DEV_GPU_UTIL"},
                    {"expr": "DCGM_FI_DEV_GPU_TEMP"},
                    {"expr": "DCGM_FI_DEV_XID_ERRORS"},
                ],
            },
            {
                "id": 4,
                "title": "RequestControl Tokens/s, 429 Shedding & Fallback Rate",
                "type": "timeseries",
                "targets": [
                    {"expr": "sum(rate(gateway_tokens_total[1m])) by (tenant_id, pool)"},
                    {"expr": "sum(rate(gateway_429_total[1m])) by (tenant_id, reason)"},
                    {"expr": "sum(rate(gateway_fallback_total[1m])) by (from_pool, to_pool)"},
                ],
            },
            {
                "id": 5,
                "title": "AIOps Anomaly Score & Remediation Actions",
                "type": "timeseries",
                "targets": [
                    {"expr": "aiops_anomaly_score"},
                    {
                        "expr": (
                            "sum(increase(aiops_remediation_actions_total[15m])) "
                            "by (kind, outcome)"
                        )
                    },
                ],
            },
        ],
    }

    return {
        "prometheus-alerts.yaml": yaml.safe_dump(alerts_doc, sort_keys=False),
        "prometheus-scrape.yaml": yaml.safe_dump(scrape_doc, sort_keys=False),
        "dcgm-counters.csv": _DCGM_COUNTERS_CSV,
        "dcgm-daemonset.yaml": yaml.safe_dump_all(dcgm_k8s_docs, sort_keys=False),
        "grafana-dashboard.json": json.dumps(dashboard_doc, indent=2),
    }
