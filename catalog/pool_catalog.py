from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


def model_slug(model_id: str) -> str:
    """Convert a Hugging Face model ID into a safe local NVMe directory slug."""
    return model_id.strip().lower().replace("/", "--")


class CatalogValidationError(ValueError):
    """Raised when a model pool catalog fails schema or invariant validation."""


class IsolationMode(StrEnum):
    """Pool tenancy and scaling isolation mode."""

    SHARED = "shared"
    DEDICATED = "dedicated"
    FALLBACK = "fallback"


class LoraAdapterSpec(BaseModel):
    """Specification for a LoRA adapter multiplexed onto a shared base pool."""

    name: str
    tenant_id: str | None = None
    s3_uri: str
    rank: int = Field(default=16, ge=1)
    dynamic: bool = True


class ModelPoolSpec(BaseModel):
    """Validated specification for a single inference model pool."""

    name: str
    model_id: str
    engine: str = "vllm"
    quantization: str = "fp16"
    isolation: IsolationMode = IsolationMode.SHARED
    fallback_pool: str | None = None
    prefetch_uri: str | None = None
    warmup_prompt: str = "Hello"
    max_model_len: int = Field(default=8192, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    enable_prefix_caching: bool = True
    isolate_prefix_cache: bool = True
    prefix_caching_hash_algo: Literal["sha256", "sha256_cbor"] = "sha256"
    enable_lora: bool = False
    max_loras: int = Field(default=4, ge=1)
    max_lora_rank: int = Field(default=16, ge=1)
    max_cpu_loras: int = Field(default=16, ge=1)
    lora_dtype: str = "auto"
    fully_sharded_loras: bool = False
    loras: list[LoraAdapterSpec] = Field(default_factory=list)
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
        if self.loras and not self.enable_lora:
            raise ValueError("loras requires enable_lora=true")
        if self.enable_lora:
            if self.max_cpu_loras < self.max_loras:
                raise ValueError("max_cpu_loras must be >= max_loras")
            for lora in self.loras:
                if lora.rank > self.max_lora_rank:
                    raise ValueError(
                        f"LoRA '{lora.name}' rank {lora.rank} > max_lora_rank {self.max_lora_rank}"
                    )
        return self


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """Resolved model pool and optional LoRA adapter for an incoming model string."""

    pool: ModelPoolSpec
    lora: LoraAdapterSpec | None
    engine_model: str


class ModelPoolCatalog:
    """Validated single source of truth for model pools and downstream target configs."""

    def __init__(
        self,
        pools: dict[str, ModelPoolSpec],
        aliases: dict[str, str],
        loras: dict[str, tuple[str, LoraAdapterSpec]] | None = None,
    ) -> None:
        self.pools = pools
        self._aliases = aliases
        self._loras = loras or {}

    def resolve_alias(self, model_or_alias: str) -> ModelPoolSpec:
        """Resolve a caller-requested model alias or pool name to its pool specification."""
        if model_or_alias in self.pools:
            return self.pools[model_or_alias]
        pool_name = self._aliases.get(model_or_alias)
        if pool_name is None:
            raise KeyError(f"Unknown model alias or pool: {model_or_alias}")
        return self.pools[pool_name]

    def resolve_target(self, model_or_alias: str) -> ResolvedTarget:
        """Resolve a model string to its backing pool, optional LoRA adapter, and engine model."""
        if model_or_alias in self._loras:
            pool_name, lora = self._loras[model_or_alias]
            return ResolvedTarget(pool=self.pools[pool_name], lora=lora, engine_model=lora.name)
        pool = self.resolve_alias(model_or_alias)
        return ResolvedTarget(pool=pool, lora=None, engine_model=pool.model_id)

    def render_targets(self) -> dict[str, Any]:
        """Render target configs for RequestControl, KEDA, Rollouts, and NVMe prefetch."""
        request_control_fallbacks = {
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
        prefetch_targets = {
            name: {
                "model_id": spec.model_id,
                "slug": model_slug(spec.model_id),
                "prefetch_uri": spec.prefetch_uri or f"s3://models/{spec.model_id}",
                "warmup_prompt": spec.warmup_prompt,
                "loras": [
                    {
                        "name": lora.name,
                        "s3_uri": lora.s3_uri,
                        "local_path": f"/models/loras/{name}/{lora.name}",
                        "tenant_id": lora.tenant_id,
                        "dynamic": lora.dynamic,
                    }
                    for lora in spec.loras
                ],
            }
            for name, spec in self.pools.items()
        }
        vllm_args = {
            name: _build_vllm_args(spec)
            for name, spec in self.pools.items()
            if spec.engine == "vllm"
        }
        return {
            "request_control_aliases": dict(self._aliases),
            "request_control_fallbacks": request_control_fallbacks,
            "keda_triggers": keda_triggers,
            "rollout_slos": rollout_slos,
            "prefetch_targets": prefetch_targets,
            "vllm_args": vllm_args,
        }


def _build_vllm_args(spec: ModelPoolSpec) -> list[str]:
    args = [
        "--model",
        spec.model_id,
        "--served-model-name",
        spec.model_id,
        "--dtype",
        "auto",
        "--max-model-len",
        str(spec.max_model_len),
        "--gpu-memory-utilization",
        f"{spec.gpu_memory_utilization:.2f}",
    ]
    if spec.quantization not in {"fp16", "bf16", "auto"}:
        args.extend(["--quantization", spec.quantization])
    if spec.enable_prefix_caching:
        args.extend(
            ["--enable-prefix-caching", "--prefix-caching-hash-algo", spec.prefix_caching_hash_algo]
        )
    if spec.enable_lora:
        args.extend(
            [
                "--enable-lora",
                "--max-loras",
                str(spec.max_loras),
                "--max-lora-rank",
                str(spec.max_lora_rank),
                "--max-cpu-loras",
                str(spec.max_cpu_loras),
                "--lora-dtype",
                spec.lora_dtype,
            ]
        )
        if spec.fully_sharded_loras:
            args.append("--fully-sharded-loras")
        static_loras = [lora for lora in spec.loras if not lora.dynamic]
        if static_loras:
            args.append("--lora-modules")
            args.extend(
                f"{lora.name}=/models/loras/{spec.name}/{lora.name}" for lora in static_loras
            )
    return args


def load_and_validate(catalog_yaml: str) -> ModelPoolCatalog:
    """Parse and validate a model pool catalog YAML document."""
    raw = yaml.safe_load(catalog_yaml)
    if not isinstance(raw, dict) or not isinstance(raw.get("pools"), dict):
        raise CatalogValidationError("Catalog must define a top-level 'pools' mapping.")

    pools: dict[str, ModelPoolSpec] = {}
    aliases: dict[str, str] = {}
    loras: dict[str, tuple[str, LoraAdapterSpec]] = {}
    for pool_name, pool_data in raw["pools"].items():
        if not isinstance(pool_data, dict):
            raise CatalogValidationError(f"Pool '{pool_name}' must be a mapping.")
        try:
            spec = ModelPoolSpec(name=pool_name, **pool_data)
        except ValidationError as exc:
            raise CatalogValidationError(f"Invalid pool '{pool_name}': {exc}") from exc
        pools[pool_name] = spec
        for alias in spec.aliases:
            if alias in aliases or alias in loras:
                existing = aliases.get(alias) or loras[alias][0]
                raise CatalogValidationError(
                    f"Duplicate alias '{alias}' mapped to '{existing}' and '{pool_name}'."
                )
            aliases[alias] = pool_name
        for lora in spec.loras:
            if lora.name in loras or lora.name in aliases or lora.name in pools:
                raise CatalogValidationError(
                    f"Duplicate LoRA adapter name '{lora.name}' in pool '{pool_name}'."
                )
            loras[lora.name] = (pool_name, lora)

    _validate_fallback_graph(pools)
    return ModelPoolCatalog(pools=pools, aliases=aliases, loras=loras)


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
