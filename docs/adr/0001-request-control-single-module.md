# 0001: Collapse Gateway, Admission, and Router into Request Control

- **Status**: Accepted
- **Context**: Splitting auth/token-reservation (for example via Envoy `ext-authz`) from in-process priority queuing, prefix routing, and SSE streaming forces token reservations, client-disconnect aborts, and post-stream token reconciliation to leak across process boundaries.
- **Decision**: Consolidate authentication, token-budget enforcement, priority admission, prefix-cache routing, fallback, and SSE metering into one Starlette/FastAPI `RequestControl` module behind an `InferenceEngine` seam.
- **Alternatives**: Envoy `ext-authz` + separate Python router; LiteLLM proxy with callbacks. Both scatter token reservation and abort propagation across extra hops.
- **Consequences**: Request lifecycle state stays local to one module and is testable end-to-end through one HTTP interface using `FakeEngineAdapter`.
- **Evidence**: Candidate 1 in [architecture-review-20261124-234524.html](../architecture-review-20261124-234524.html).
