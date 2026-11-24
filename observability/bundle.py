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


def render_observability_bundle(catalog: ModelPoolCatalog) -> dict[str, str]:
    """Render Prometheus alerts, scrape config, DCGM CSV, and Grafana dashboard JSON."""
    pool_rules: list[dict[str, Any]] = []
    for name, spec in catalog.pools.items():
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
                    "expr": (
                        f'(sum(rate(gateway_requests_total{{pool="{name}",status=~"5.."}}[5m])) '
                        f'/ clamp_min(sum(rate(gateway_requests_total{{pool="{name}"}}[5m])), '
                        f'1e-6)) > {14.4 * err_budget:.6f} '
                        f'and '
                        f'(sum(rate(gateway_requests_total{{pool="{name}",status=~"5.."}}[1h])) '
                        f'/ clamp_min(sum(rate(gateway_requests_total{{pool="{name}"}}[1h])), '
                        f'1e-6)) > {14.4 * err_budget:.6f}'
                    ),
                    "labels": {"severity": "critical", "pool": name},
                    "annotations": {
                        "summary": f"Fast error-budget burn rate on {name} (14.4x over 5m & 1h)",
                    },
                },
                {
                    "alert": f"PoolErrorBudgetSlowBurnRate_{name}",
                    "expr": (
                        f'(sum(rate(gateway_requests_total{{pool="{name}",status=~"5.."}}[30m])) '
                        f'/ clamp_min(sum(rate(gateway_requests_total{{pool="{name}"}}[30m])), '
                        f'1e-6)) > {6.0 * err_budget:.6f} '
                        f'and '
                        f'(sum(rate(gateway_requests_total{{pool="{name}",status=~"5.."}}[6h])) '
                        f'/ clamp_min(sum(rate(gateway_requests_total{{pool="{name}"}}[6h])), '
                        f'1e-6)) > {6.0 * err_budget:.6f}'
                    ),
                    "labels": {"severity": "warning", "pool": name},
                    "annotations": {
                        "summary": f"Slow error-budget burn rate on {name} (6x over 30m & 6h)",
                    },
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
        {
            "alert": "VLLMKVCacheSaturated",
            "expr": "vllm:kv_cache_usage_perc > 0.95",
            "for": "2m",
            "labels": {"severity": "warning"},
            "annotations": {"summary": "vLLM KV cache usage above 95%"},
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
        "scrape_configs": [
            {
                "job_name": "request-control",
                "static_configs": [{"targets": ["request-control:8080"]}],
            },
            {
                "job_name": "vllm-engines",
                "static_configs": [{"targets": ["vllm-pool-8b:8000", "vllm-pool-fallback:8000"]}],
            },
            {
                "job_name": "dcgm-exporter",
                "static_configs": [{"targets": ["dcgm-exporter:9400"]}],
            },
        ],
    }
    dashboard_doc = {
        "title": "Multi-Tenant LLM Serving Platform",
        "schemaVersion": 39,
        "panels": [
            {
                "id": 1,
                "title": "p95 TTFT by Pool (vLLM & Gateway)",
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
                "title": "Gateway Tokens/s, 429 Shedding & Fallback Rate",
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
        "grafana-dashboard.json": json.dumps(dashboard_doc, indent=2),
    }
