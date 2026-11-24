# 0002: Separate Fast-Path Routing from Slow-Path Remediation via Pool Health

- **Status**: Accepted
- **Context**: Having both the request router and the Kubernetes remediation loop shift pool fallback and edit `max-num-seqs` on Deployments causes control-loop flapping and conflicts with ArgoCD reconciliation.
- **Decision**: Give `RequestControl` sole ownership of fast-path request fallback and low-priority load shedding, and confine `SelfHealingEngine` to slow-path infrastructure actions (`cordon`, `drain`, `restart`) plus setting time-bounded `PoolHealth` overrides rather than mutating ArgoCD-managed Deployment arguments.
- **Alternatives**: Letting a Kubernetes operator patch Deployment args and Gateway config maps directly; rejected because ArgoCD overwrites out-of-band Deployment edits.
- **Consequences**: Eliminates fallback oscillation and GitOps drift while keeping a single coordination seam (`PoolHealth`) between data and control planes.
- **Evidence**: Candidate 2 in [architecture-review-20261124-234524.html](../architecture-review-20261124-234524.html).
