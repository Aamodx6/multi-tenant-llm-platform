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
    assert "--quantization" in targets["vllm_args"]["pool-8b"]
    assert "awq" in targets["vllm_args"]["pool-8b"]
    assert "--enable-prefix-caching" in targets["vllm_args"]["pool-8b"]
    assert "--quantization" not in targets["vllm_args"]["pool-fallback"]


def test_lora_catalog_validation_and_vllm_flags() -> None:
    lora_yaml = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    enable_lora: true
    max_loras: 4
    max_lora_rank: 16
    max_cpu_loras: 8
    fully_sharded_loras: true
    min_replicas: 1
    max_replicas: 4
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.80
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    loras:
      - name: tenant-a-sql
        tenant_id: tenant-a
        s3_uri: s3://models/loras/tenant-a-sql
        rank: 16
        dynamic: true
      - name: shared-json
        s3_uri: s3://models/loras/shared-json
        rank: 8
        dynamic: false
"""
    cat = load_and_validate(lora_yaml)
    resolved = cat.resolve_target("tenant-a-sql")
    assert resolved.pool.name == "pool-8b"
    assert resolved.lora is not None
    assert resolved.lora.tenant_id == "tenant-a"
    assert resolved.engine_model == "tenant-a-sql"

    targets = cat.render_targets()
    args = targets["vllm_args"]["pool-8b"]
    assert "--enable-lora" in args
    assert "--max-loras" in args and "4" in args
    assert "--max-cpu-loras" in args and "8" in args
    assert "--fully-sharded-loras" in args
    assert "--lora-modules" in args
    assert "shared-json=/models/loras/pool-8b/shared-json" in args
    assert any(
        item["name"] == "tenant-a-sql"
        for item in targets["prefetch_targets"]["pool-8b"]["loras"]
    )

    bad_cpu = lora_yaml.replace("max_cpu_loras: 8", "max_cpu_loras: 2")
    with pytest.raises(CatalogValidationError, match="max_cpu_loras"):
        load_and_validate(bad_cpu)

    bad_rank = lora_yaml.replace("        rank: 16", "        rank: 32")
    with pytest.raises(CatalogValidationError, match="rank"):
        load_and_validate(bad_rank)


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


def test_prefix_caching_emits_sha256_hash_algo_and_isolation_flag() -> None:
    yaml_doc = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    enable_prefix_caching: true
    isolate_prefix_cache: true
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    cat = load_and_validate(yaml_doc)
    args = cat.render_targets()["vllm_args"]["pool-8b"]
    idx = args.index("--prefix-caching-hash-algo")
    assert args[idx + 1] == "sha256"
    assert cat.pools["pool-8b"].isolate_prefix_cache is True


def test_rejects_path_traversal_and_shell_metacharacters_in_catalog() -> None:
    bad_lora_name = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    enable_lora: true
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
    loras:
      - name: ../../other-pool/secret
        s3_uri: s3://models/loras/good
"""
    with pytest.raises(CatalogValidationError):
        load_and_validate(bad_lora_name)

    bad_s3_uri = """
pools:
  pool-8b:
    model_id: Qwen/Qwen2.5-7B-Instruct
    prefetch_uri: "s3://models/qwen; rm -rf /"
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError):
        load_and_validate(bad_s3_uri)

    bad_pool_name = """
pools:
  "pool$(id)":
    model_id: Qwen/Qwen2.5-7B-Instruct
    min_replicas: 1
    max_replicas: 2
    keda_waiting_threshold: 4
    keda_kv_cache_threshold: 0.8
    slo_p95_ttft_ms: 1000
    slo_availability: 0.995
"""
    with pytest.raises(CatalogValidationError):
        load_and_validate(bad_pool_name)
