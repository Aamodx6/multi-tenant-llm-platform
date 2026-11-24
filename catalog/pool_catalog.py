from __future__ import annotations

from enum import StrEnum
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class CatalogValidationError(ValueError):
    """Raised when a model pool catalog fails schema or invariant validation."""


class IsolationMode(StrEnum):
    """Pool tenancy and scaling isolation mode."""

    SHARED = "shared"
    DEDICATED = "dedicated"
    FALLBACK = "fallback"


class ModelPoolSpec(BaseModel):
    """Validated specification for a single inference model pool."""

    name: str
    model_id: str
    engine: str = "vllm"
    quantization: str = "fp16"
    isolation: IsolationMode = IsolationMode.SHARED
    fallback_pool: str | None = None
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    keda_waiting_threshold: int = Field(gt=0)
    keda_kv_cache_threshold: float = Field(gt=0.0, le=1.0)
    slo_p95_ttft_ms: int = Field(gt=0)
    slo_availability: float = Field(gt=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_pool_invariants(self) -> ModelPoolSpec:
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")
        if self.isolation == IsolationMode.SHARED and self.min_replicas == 0:
            raise ValueError("shared pool cannot enable scale-to-zero (min_replicas=0)")
        return self


class ValidatedCatalog:
    """Validated single source of truth for model pools and downstream target configs."""

    def __init__(self, pools: dict[str, ModelPoolSpec], aliases: dict[str, str]) -> None:
        self.pools = pools
        self._aliases = aliases

    def resolve_alias(self, model_or_alias: str) -> ModelPoolSpec:
        """Resolve a caller-requested model alias or pool name to its pool specification."""
        if model_or_alias in self.pools:
            return self.pools[model_or_alias]
        pool_name = self._aliases.get(model_or_alias)
        if pool_name is None:
            raise KeyError(f"Unknown model alias or pool: {model_or_alias}")
        return self.pools[pool_name]

    def render_targets(self) -> dict[str, Any]:
        """Render target configurations for RequestControl, KEDA, and Argo Rollouts."""
        gateway_fallbacks = {
            name: spec.fallback_pool
            for name, spec in self.pools.items()
            if spec.fallback_pool is not None
        }
        keda_triggers = {
            name: {
                "min_replicas": spec.min_replicas,
                "max_replicas": spec.max_replicas,
                "waiting_threshold": spec.keda_waiting_threshold,
                "kv_cache_threshold": spec.keda_kv_cache_threshold,
            }
            for name, spec in self.pools.items()
        }
        rollout_slos = {
            name: {
                "p95_ttft_ms": spec.slo_p95_ttft_ms,
                "availability": spec.slo_availability,
            }
            for name, spec in self.pools.items()
        }
        return {
            "gateway_aliases": dict(self._aliases),
            "gateway_fallbacks": gateway_fallbacks,
            "keda_triggers": keda_triggers,
            "rollout_slos": rollout_slos,
        }


def load_and_validate(catalog_yaml: str) -> ValidatedCatalog:
    """Parse and validate a model pool catalog YAML document."""
    raw = yaml.safe_load(catalog_yaml)
    if not isinstance(raw, dict) or not isinstance(raw.get("pools"), dict):
        raise CatalogValidationError("Catalog must define a top-level 'pools' mapping.")

    pools: dict[str, ModelPoolSpec] = {}
    aliases: dict[str, str] = {}
    for pool_name, pool_data in raw["pools"].items():
        if not isinstance(pool_data, dict):
            raise CatalogValidationError(f"Pool '{pool_name}' must be a mapping.")
        try:
            spec = ModelPoolSpec(name=pool_name, **pool_data)
        except ValidationError as exc:
            raise CatalogValidationError(f"Invalid pool '{pool_name}': {exc}") from exc
        pools[pool_name] = spec
        for alias in spec.aliases:
            if alias in aliases:
                raise CatalogValidationError(
                    f"Duplicate alias '{alias}' mapped to '{aliases[alias]}' and '{pool_name}'."
                )
            aliases[alias] = pool_name

    _validate_fallback_graph(pools)
    return ValidatedCatalog(pools=pools, aliases=aliases)


def _validate_fallback_graph(pools: dict[str, ModelPoolSpec]) -> None:
    """Verify every fallback_pool exists and the fallback graph is acyclic."""
    for name, spec in pools.items():
        if spec.fallback_pool is not None and spec.fallback_pool not in pools:
            raise CatalogValidationError(
                f"Pool '{name}' references unknown fallback_pool '{spec.fallback_pool}'."
            )

    for start in pools:
        visited: set[str] = set()
        curr: str | None = start
        while curr is not None:
            if curr in visited:
                raise CatalogValidationError(f"Fallback cycle detected involving pool '{curr}'.")
            visited.add(curr)
            curr = pools[curr].fallback_pool
