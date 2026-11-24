# 0003: Encapsulate Redis Outage Degradation in Tenant Budget and Keep Queues Local

- **Status**: Accepted
- **Context**: Coupling API key caching, token buckets, admission queues, and router state to Redis leaks outage-degradation logic across four callers and risks stalling local request queuing if Redis becomes unreachable.
- **Decision**: Keep priority admission queues and engine load tracking strictly in-process per replica, and encapsulate Postgres lookup, Redis token-bucket math, and conservative local fallback inside one `TenantBudget` module.
- **Alternatives**: Backing admission queues with Redis lists; rejected because Redis unavailability would block local admission shedding.
- **Consequences**: Redis failure behavior is implemented and tested in one place, and admission queues continue enforcing wait budgets during Redis degradation.
- **Evidence**: Candidate 3 in [architecture-review-20261124-234524.html](../architecture-review-20261124-234524.html).
