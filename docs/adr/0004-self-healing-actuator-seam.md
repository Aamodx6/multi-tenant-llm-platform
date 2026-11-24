# 0004: Co-Locate Anomaly Scoring and Remediation Policy Behind Cluster Actuator

- **Status**: Accepted
- **Context**: Routing anomaly scores from a detector through Prometheus metrics and Alertmanager webhooks to a separate remediation controller adds scrape latency to MTTD and prevents deterministic testing of escalation and guardrails.
- **Decision**: Combine rolling statistical anomaly scoring (z-score and baseline threshold comparison) with `RemediationPolicy` guardrail evaluation inside one pure `SelfHealingEngine` behind a `ClusterActuator` seam (`K8sActuatorAdapter` in production, `SimulatedActuatorAdapter` in tests).
- **Alternatives**: Async Prometheus Alertmanager webhook pipeline; rejected as the primary control trigger, though anomaly scores are still exported to Prometheus for dashboards.
- **Consequences**: Entire detection-to-remediation escalation chains (cooldowns, max-one-node drain, two-failure human paging) are unit-testable in milliseconds without a live cluster.
- **Evidence**: Candidate 4 in [architecture-review-20261124-234524.html](../architecture-review-20261124-234524.html).
