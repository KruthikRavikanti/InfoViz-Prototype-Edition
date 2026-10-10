#!/usr/bin/env python3
"""Verify a staged model expansion and retain a recoverable baseline snapshot.

This is a release-boundary check, not an inference or model-substitution tool.
It does not install data or change the running applications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "artifacts/coverage-expansion/baseline"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def child(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"Release artifact path must be relative: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError(f"Unsafe or missing release artifact: {path}")
    return path


def snapshot_baseline(repo: Path = REPO_ROOT, destination: Path = BASELINE) -> dict:
    """Publish a full byte-verified snapshot without replacing a prior snapshot."""
    locations = ("artifacts/reference", "public/data", "data/image_by_model")
    if destination.exists():
        report = document(destination / "snapshot.json")
        for relative, expected in report["files"].items():
            if digest(child(destination, relative)) != expected:
                raise ValueError(f"Existing baseline snapshot is corrupt: {relative}")
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".baseline-snapshot-", dir=destination.parent))
    try:
        hashes = {}
        for relative in locations:
            source = repo / relative
            shutil.copytree(source, temporary / relative)
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                name = path.relative_to(repo).as_posix()
                if path.is_symlink():
                    raise ValueError(f"Baseline unexpectedly contains a symlink: {name}")
                hashes[name] = digest(path)
                if digest(temporary / name) != hashes[name]:
                    raise ValueError(f"Baseline copy failed its checksum: {name}")
        reference = document(repo / "artifacts/reference/manifest.json")
        report = {"reference_version": reference["reference_version"],
                  "models": reference["supported_models"], "files": hashes}
        (temporary / "snapshot.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise ValueError("Baseline snapshot destination appeared; refusing replacement")
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_inference_provenance(reference: dict, raw_root: Path, core_root: Path) -> None:
    """Recheck inference proofs at the release boundary, not just CSV hashes."""
    from scripts.data.extend_reference_coverage import (
        precision_policy_from_provenance, prediction_policy_from_provenance,
    )
    extension = reference["coverage_extension"]
    added = set(extension["added_models"])
    for name in ("new_inference_provenance_sha256", "model_prediction_policies"):
        if set(extension.get(name, {})) != added:
            raise ValueError(f"Inference provenance roster is incomplete: {name}")
    precision = extension.get("prediction_policies")
    if precision is not None and set(precision) != added:
        raise ValueError("Inference numerical-policy roster is incomplete")
    for model in sorted(added):
        path = child(raw_root / model, "provenance.json")
        if digest(path) != extension["new_inference_provenance_sha256"][model]:
            raise ValueError(f"Inference provenance changed: {model}")
        proof = document(path)
        if prediction_policy_from_provenance(model, proof) != extension["model_prediction_policies"][model]:
            raise ValueError(f"Inference noise policy changed: {model}")
        if precision is not None:
            policy = precision_policy_from_provenance(model, proof)
            if (policy != precision[model]
                    or policy["precision_helper_sha256"] != digest(core_root / "cortex/utils/inference_precision.py")):
                raise ValueError(f"Inference numerical policy or helper source changed: {model}")


def verify_release(stage: Path, spatial_stage: Path, *, repo: Path = REPO_ROOT,
                   baseline: Path = BASELINE, raw_root: Path | None = None) -> dict:
    raw_root = raw_root or repo / "artifacts/coverage-expansion/raw"
    artifact_root = stage / "artifacts/reference"
    data_root = stage / "public/data"
    brain_root = spatial_stage / "public/data/brain/reference"
    atlas_root = spatial_stage / "public/data/voxel_selectivity_atlas"
    reference = document(artifact_root / "manifest.json")
    public = document(data_root / "reference_manifest.json")
    report = document(stage / "preservation_report.json")
    original = document(baseline / "artifacts/reference/manifest.json")
    version = reference["reference_version"]
    models = reference["supported_models"]
    baseline_models = set(original["supported_models"])
    added = set(models) - baseline_models
    if (not re.fullmatch(r"[a-f0-9]{64}", version) or version == original["reference_version"]
            or models != sorted(set(models)) or not added or not baseline_models < set(models)
            or report["reference_version"] != version
            or reference["coverage_extension"]["parent_reference_version"] != original["reference_version"]):
        raise ValueError("Release identity does not extend the preserved baseline")

    sys.path.insert(0, str(repo.parent / "cortex-core"))
    sys.path.insert(0, str(repo))
    from cortex.comparison.artifacts import ArtifactRepository
    from cortex.comparison.brain_map_audit import validate_spatial_audit_extension
    from scripts.data import build_reference_artifacts as builder

    verify_inference_provenance(reference, raw_root, repo.parent / "cortex-core")
    if public != builder.public_manifest_from_full(reference):
        raise ValueError("Public/private reference identities disagree")
    snapshot = document(baseline / "snapshot.json")
    rewritten = {
        "artifacts/reference/manifest.json", "public/data/combined_image_means.csv",
        "public/data/reference_manifest.json",
        *[f"public/data/{directory}/voxel_cluster_summary.{suffix}"
          for directory in ("voxel_clusters", "voxel_clusters_hierarchical_corr")
          for suffix in ("csv", "json")],
    }
    for name, hashes in (("artifacts/reference", report["private_artifact_sha256"]),
                         ("public/data", report["public_artifact_sha256"])):
        expected_paths = {path[len(name) + 1:] for path in snapshot["files"]
                          if path.startswith(name + "/") and path not in rewritten}
        if set(hashes) != expected_paths:
            raise ValueError(f"Baseline preservation report has incomplete coverage: {name}")
        for relative, expected in hashes.items():
            if digest(child(stage / name, relative)) != expected:
                raise ValueError(f"A staged baseline artifact changed: {name}/{relative}")
            if digest(child(baseline / name, relative)) != expected:
                raise ValueError(f"Baseline preservation report disagrees: {name}/{relative}")

    repository = ArtifactRepository(artifact_root)
    if repository.capabilities()["supported_models"] != models:
        raise ValueError("Core capability roster differs from the release")
    combined_path = data_root / "combined_image_means.csv"
    if digest(combined_path) != reference["combined_means"]["sha256"]:
        raise ValueError("Plotted aggregate response CSV failed its checksum")
    combined = builder.load_combined_means(combined_path, models)
    sources = {}
    for model in models:
        if model in baseline_models and reference["artifacts"][model] != original["artifacts"][model]:
            raise ValueError(f"Baseline artifact metadata changed: {model}")
        for roi in reference["rois"]:
            artifact = repository.load(model, roi)
            entry = reference["artifacts"][model][roi]
            source_root = raw_root / model if model in added else baseline / "data/image_by_model"
            source_path = child(source_root, entry["source_file"])
            if digest(source_path) != entry["source_sha256"]:
                raise ValueError(f"Raw prediction source changed: {model}/{roi}")
            key = f"{model}/{roi}"
            if model in baseline_models and reference["source_selection"][key] != original["source_selection"][key]:
                raise ValueError(f"Baseline source selection changed: {key}")
            source = builder.load_and_validate_source(
                builder.SourceCandidate(model, roi, entry["source_timestamp"], source_path),
                expected_ids=builder.canonical_image_ids(), expected_voxel_count=builder.VOXEL_COUNTS[roi],
                expected_means=combined[f"{model}_{roi}"].to_numpy(dtype="float64"),
            )
            import numpy as np
            if not np.array_equal(artifact.reference_voxels, source.voxels.astype(np.float32)):
                raise ValueError(f"Core raw matrix differs from the plotted source: {key}")
            sources[(model, roi)] = source
    contracts = builder.load_static_voxel_cluster_contracts(
        sources, kmeans_dir=data_root / "voxel_clusters",
        hierarchical_dir=data_root / "voxel_clusters_hierarchical_corr",
    )
    visual = reference["optional_artifacts"]["visual"]
    static_sha = builder.sha256_json({
        "voxel": {f"{model}/{roi}": {"kmeans": contract.kmeans_sha256,
                                     "hierarchical": contract.hierarchical_sha256}
                  for (model, roi), contract in sorted(contracts.items())},
        "visual": {"kmeans": visual["static_kmeans_sha256"],
                   "hierarchical": visual["static_hierarchical_sha256"]},
    })
    identity = builder._reference_identity(
        models=models, image_ids=builder.canonical_image_ids(),
        combined_means_sha256=digest(combined_path), selected=sources,
        visual_features_sha256=visual["source_sha256"], visual_images_sha256=visual["source_images_sha256"],
        static_clusters_sha256=static_sha,
    )
    identity["coverage_extension"] = reference["coverage_extension"]
    if builder.sha256_json(identity) != version:
        raise ValueError("Full release identity does not reproduce its SHA-256 version")

    brain = document(brain_root / "manifest.json")
    atlas = document(atlas_root / "manifest.json")
    audit = document(brain_root / "spatial-audit.json")
    if (brain["referenceVersion"] != version or atlas["referenceVersion"] != version
            or set(brain["models"]) != set(models) or atlas["supportedModels"] != models
            or set(atlas["models"]) != set(models)
            or set(validate_spatial_audit_extension(audit, reference_version=version)) != set(models)):
        raise ValueError("Reference, brain, atlas, and spatial-audit identities disagree")
    for model, entry in brain["models"].items():
        path = child(brain_root, entry["url"])
        if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Brain payload does not verify: {model}")
    for model, entry in atlas["models"].items():
        for kind in ("points", "summary"):
            path = child(atlas_root, entry[f"{kind}Url"])
            if digest(path) != entry[f"{kind}Sha256"] or path.stat().st_size != entry[f"{kind}Bytes"]:
                raise ValueError(f"Atlas payload does not verify: {model}/{kind}")
    for relative_root in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
        for path in (baseline / relative_root).rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                relative = path.relative_to(baseline)
                if digest(child(spatial_stage, relative.as_posix())) != digest(path):
                    raise ValueError(f"Spatial expansion changed a baseline child: {relative}")
    scores = document(repo / "public/data/murty185_zero_index.json")
    scored = set(scores["ffa"]) - {"ceiling"}
    if not set(models).issubset(scored):
        raise ValueError("Release includes non-inferable/unscored models")
    return {
        "reference_version": version, "baseline_models": len(baseline_models),
        "model_count": len(models), "roi_count": len(reference["rois"]),
        "reference_count": reference["reference_count"], "added_models": sorted(added),
        "remaining_scoreboard_models": sorted(scored - set(models)),
        "brain_manifest_sha256": digest(brain_root / "manifest.json"),
        "atlas_manifest_sha256": digest(atlas_root / "manifest.json"),
        "spatial_audit_sha256": digest(brain_root / "spatial-audit.json"),
        "baseline_preserved": True, "live_install_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-baseline", action="store_true")
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--spatial-stage-root", type=Path)
    parser.add_argument("--raw-root", type=Path,
                        help="Exact prediction staging directory for this candidate (for example raw-fp32)")
    args = parser.parse_args()
    if args.snapshot_baseline:
        report = snapshot_baseline()
        print(json.dumps({"baseline_version": report["reference_version"], "verified_files": len(report["files"])}))
    elif args.stage_root and args.spatial_stage_root:
        print(json.dumps(verify_release(args.stage_root, args.spatial_stage_root,
                                        raw_root=args.raw_root), sort_keys=True, indent=2))
    else:
        parser.error("Choose --snapshot-baseline or both staged release roots")


if __name__ == "__main__":
    main()
