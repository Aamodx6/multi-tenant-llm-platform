import pytest

from catalog.pool_catalog import CatalogValidationError, load_and_validate

VALID_CATALOG_YAML = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    engine: vllm
    quantization: awq
    isolation: shared
    fallback_pool: pool-fallback
    min_replicas: 1
    max_replicas: 6
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases:
      - qwen-7b
      - default
  pool-fallback:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    engine: vllm
    quantization: fp16
    isolation: fallback
    min_replicas: 0
    max_replicas: 4
    keda_waiting_threshold: 6
    keda_kv_cache_threshold: 0.85
    slo_p95_ttft_ms: 600
    slo_availability: 0.99
    aliases:
      - qwen-1.5b
"""


def test_load_and_validate_renders_gateway_keda_and_rollouts_targets() -> None:
    catalog = load_and_validate(VALID_CATALOG_YAML)

    assert catalog.resolve_alias("qwen-7b").name == "pool-8b"
    assert catalog.resolve_alias("default").fallback_pool == "pool-fallback"

    targets = catalog.render_targets()
    assert targets["request_control_aliases"]["qwen-7b"] == "pool-8b"
    assert targets["request_control_fallbacks"]["pool-8b"] == "pool-fallback"
    assert targets["keda_triggers"]["pool-8b"]["min_replicas"] == 1
    assert targets["keda_triggers"]["pool-8b"]["waiting_threshold"] == 4
    assert targets["rollout_slos"]["pool-8b"]["p95_ttft_ms"] == 1000
    assert targets["rollout_slos"]["pool-fallback"]["availability"] == 0.99
    assert targets["prefetch_targets"]["pool-8b"]["prefetch_uri"] == "s3://models/Qwen/Qwen2.5-7B-Instruct"
    assert targets["prefetch_targets"]["pool-8b"]["warmup_prompt"] == "Hello"


def test_rejects_unknown_fallback_pool() -> None:
    bad_yaml = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    fallback_pool: nonexistent
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError, match="fallback_pool"):
        load_and_validate(bad_yaml)


def test_rejects_fallback_cycle() -> None:
    bad_yaml = """
pools:
  pool-a:
    model_id: model-a
    fallback_pool: pool-b
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
  pool-b:
    model_id: model-b
    fallback_pool: pool-a
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError, match="cycle"):
        load_and_validate(bad_yaml)


def test_rejects_shared_pool_with_scale_to_zero() -> None:
    bad_yaml = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    isolation: shared
    min_replicas: 0
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError, match="scale-to-zero"):
        load_and_validate(bad_yaml)


def test_rejects_min_replicas_exceeding_max_and_duplicate_aliases() -> None:
    bad_bounds = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    min_replicas: 5
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError, match="min_replicas"):
        load_and_validate(bad_bounds)

    dup_alias = """
pools:
  pool-a:
    model_id: m1
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases: [chat]
  pool-b:
    model_id: m2
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    aliases: [chat]
"""
    with pytest.raises(CatalogValidationError, match="Duplicate alias"):
        load_and_validate(dup_alias)
