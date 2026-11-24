# 0005: Centralize Model Pool Configuration in Model Pool Catalog

- **Status**: Accepted
- **Context**: Model pool aliases, fallback targets, engine families, NVMe cache paths, KEDA scaling bounds, and SLO thresholds were duplicated across gateway, deployment, controller, and observability configs, alongside an ambiguity between shared pools and per-tenant scale-to-zero.
- **Decision**: Introduce `ModelPoolCatalog` as a validated single source of truth that enforces pool invariants (shared vs dedicated isolation, acyclic fallback targets, positive KEDA/SLO bounds) and renders target configs for downstream consumers.
- **Alternatives**: Maintaining separate Helm values, gateway YAML, and Prometheus alert templates by hand; rejected due to config drift.
- **Consequences**: Breaking fallback references or conflicting scale-to-zero settings fail validation at CI time rather than at runtime.
- **Evidence**: Candidate 5 in [architecture-review-20261124-234524.html](../architecture-review-20261124-234524.html).
