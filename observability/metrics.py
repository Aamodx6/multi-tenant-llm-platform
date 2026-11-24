from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class MetricsRegistry:
    """Isolated Prometheus metrics registry for RequestControl and SelfHealingEngine."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "gateway_requests_total",
            "Total requests handled by RequestControl.",
            ["pool", "priority", "status"],
            registry=self.registry,
        )
        self.rate_limited_total = Counter(
            "gateway_429_total",
            "Total HTTP 429 rejections by tenant and reason.",
            ["tenant_id", "reason"],
            registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "gateway_queue_wait_seconds",
            "Priority queue admission wait duration in seconds.",
            ["priority"],
            registry=self.registry,
        )
        self.ttft_seconds = Histogram(
            "gateway_ttft_seconds",
            "Time to first token observed at RequestControl.",
            ["pool", "priority"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.fallback_total = Counter(
            "gateway_fallback_total",
            "Total requests shifted from primary pool to fallback pool.",
            ["from_pool", "to_pool"],
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "gateway_tokens_total",
            "Metered prompt + completion tokens by tenant and pool.",
            ["tenant_id", "pool"],
            registry=self.registry,
        )
        self.anomaly_score = Gauge(
            "aiops_anomaly_score",
            "Latest MAD modified z-score computed by SelfHealingEngine.",
            ["pod", "pool"],
            registry=self.registry,
        )
        self.remediation_actions_total = Counter(
            "aiops_remediation_actions_total",
            "Remediation actions evaluated by SelfHealingEngine.",
            ["kind", "outcome"],
            registry=self.registry,
        )

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def render(self) -> bytes:
        """Render Prometheus exposition bytes."""
        return generate_latest(self.registry)
