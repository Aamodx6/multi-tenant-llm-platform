from __future__ import annotations

import argparse
from pathlib import Path

from catalog.pool_catalog import load_and_validate
from deploy.manifest_renderer import DeployConfig, render_deploy_manifests, write_manifests
from observability.bundle import render_observability_bundle


def main() -> int:
    """Render Kubernetes deploy manifests and observability configs from catalog/pools.yaml."""
    parser = argparse.ArgumentParser(description="Render K8s and Observability manifests")
    parser.add_argument("--catalog", type=Path, default=Path("catalog/pools.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("deploy/generated"))
    parser.add_argument("--namespace", default="llm")
    args = parser.parse_args()

    catalog = load_and_validate(args.catalog.read_text(encoding="utf-8"))
    manifests = render_deploy_manifests(catalog, DeployConfig(namespace=args.namespace))
    obs = render_observability_bundle(catalog)
    write_manifests({**manifests, **obs}, args.out_dir)
    print(f"Wrote {len(manifests) + len(obs)} files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
