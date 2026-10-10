#!/usr/bin/env python3
"""Stage a full, explicitly authorized FP32 reference refresh; never install it.

Unlike an append-only expansion, this mode refits every existing model from
fresh inference. The complete parent release remains recoverable and shared
visual, measured, DreamSim, anatomical, and historical plot files are retained.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from . import build_reference_artifacts as reference
    from . import extend_reference_coverage as expansion
    from . import generate_model_reference_predictions as inference
    from .verify_coverage_release import child, digest, document
except ImportError:
    import build_reference_artifacts as reference  # type: ignore
    import extend_reference_coverage as expansion  # type: ignore
    import generate_model_reference_predictions as inference  # type: ignore
    from verify_coverage_release import child, digest, document  # type: ignore

REPO_ROOT = reference.REPO_ROOT
STANDARDIZATION_ROOT = REPO_ROOT / "artifacts/precision-standardization"
POLICY = "full_roster_fp32_regeneration_v1"
ANALYSIS_INPUT_POLICY = "canonical_float32_promoted_float64_v1"
PARENT_REFERENCE_VERSION = "a862bba8496384d80de1b1990152b2b683e4325a52b0880d85783072172246d0"
PARENT_MANIFEST_SHA256 = "9258d85bfb911db0a4dc959659d0e8b57e69957fc73f671416fdeea9a96c7475"
SNAPSHOT_ROOTS = ("artifacts/reference", "public/data", "data/image_by_model")
REWRITTEN_PUBLIC_PATHS = {
    "combined_image_means.csv", "reference_manifest.json",
    *[f"{directory}/voxel_cluster_summary.{suffix}"
      for directory in expansion.CLUSTER_DIRS for suffix in ("csv", "json")],
}


def validate_parent_snapshot(
    baseline_root: Path, *, parent_version: str = PARENT_REFERENCE_VERSION,
    parent_manifest_sha256: str = PARENT_MANIFEST_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a complete, byte-verified snapshot of the reviewed current release."""
    snapshot = document(child(baseline_root, "snapshot.json"))
    manifest_path = child(baseline_root, "artifacts/reference/manifest.json")
    parent = document(manifest_path)
    models = expansion.validate_baseline_manifest(parent)
    if (parent["reference_version"] != parent_version
            or digest(manifest_path) != parent_manifest_sha256
            or snapshot.get("reference_version") != parent_version
            or snapshot.get("models") != list(models)):
        raise reference.ReferenceValidationError("Refresh parent is not the explicitly reviewed release")
    actual = {path.relative_to(baseline_root).as_posix()
              for root in SNAPSHOT_ROOTS for path in (baseline_root / root).rglob("*")
              if path.is_file()}
    if set(snapshot.get("files", {})) != actual:
        raise reference.ReferenceValidationError("Refresh snapshot does not cover all parent scientific files")
    for relative, expected in snapshot["files"].items():
        if digest(child(baseline_root, relative)) != expected:
            raise reference.ReferenceValidationError(f"Refresh parent snapshot changed: {relative}")
    return parent, snapshot


def validate_refresh_roster(models: Sequence[str], parent: Mapping[str, Any], scoreboard_path: Path) -> tuple[str, ...]:
    if (not models or len(models) != len(set(models))
            or set(models) != set(parent["supported_models"])):
        raise reference.ReferenceValidationError("Full refresh must regenerate the exact existing model roster")
    target = tuple(sorted(models))
    expansion.validate_scoreboard_roster(target, scoreboard_path)
    scoreboard = document(scoreboard_path)
    if set(target) != set(scoreboard["ffa"]) - {"ceiling"}:
        raise reference.ReferenceValidationError("Full refresh must cover every runnable scoreboard model")
    return target


def validate_inference_proofs(
    *, models: Sequence[str], raw_root: Path,
    sources: Mapping[tuple[str, str], reference.ValidatedSource],
    core_root: Path, image_root: Path,
) -> dict[str, dict[str, Any]]:
    """Bind every model to current FP32 source, canonical stimuli, and raw columns."""
    helper_sha = digest(child(core_root, "cortex/utils/inference_precision.py"))
    images = [{"id": image, "sha256": digest(child(image_root, image))}
              for image in reference.canonical_image_ids()]
    proofs = {}
    current_sources = inference.core_source_identity(core_root)["core_inference_source_sha256"]
    for model in models:
        proof = document(child(raw_root / model, "provenance.json"))
        precision = expansion.precision_policy_from_provenance(model, proof)
        expansion.prediction_policy_from_provenance(model, proof)
        if (proof.get("schema_version") != 1 or proof.get("model") != model
                or proof.get("dataset") != reference.DATASET
                or proof.get("reference_count") != reference.REFERENCE_COUNT
                or proof.get("image_ids") != list(reference.canonical_image_ids())
                or proof.get("images") != images
                or proof.get("image_set_sha256") != reference.sha256_json(images)
                or proof.get("subject_order") != list(inference.SUBJECT_ORDER)
                or proof.get("subject_voxel_counts") != inference.SUBJECT_VOXEL_COUNTS
                or proof.get("voxel_counts") != reference.VOXEL_COUNTS
                or proof.get("preprocessing") != inference.PREPROCESSING
                or type(proof.get("seed")) is not int or proof["seed"] != 0
                or precision["precision_helper_sha256"] != helper_sha
                or set(proof.get("rois", {})) != set(reference.ROIS)):
            raise reference.ReferenceValidationError(f"Fresh {model} inference is not the reviewed canonical FP32 prediction space")
        if proof["core_inference_source_sha256"] != current_sources:
            raise reference.ReferenceValidationError(f"Fresh {model} inference lacks the complete current Core source identity")
        for roi in reference.ROIS:
            source = sources[(model, roi)]
            entry = proof["rois"][roi]
            if (entry.get("voxel_columns") != list(source.voxel_columns)
                    or entry.get("voxel_order_sha256") != reference.sha256_json(list(source.voxel_columns))
                    or entry.get("output_shape") != [reference.REFERENCE_COUNT, reference.VOXEL_COUNTS[roi]]):
                raise reference.ReferenceValidationError(f"Fresh {model}/{roi} proof has different voxel identity/order")
        proofs[model] = proof
    return proofs


def write_combined_means(path: Path, models: Sequence[str], sources: Mapping[tuple[str, str], reference.ValidatedSource]) -> None:
    columns = ["image_name"] + [f"{model}_{roi}" for model in models for roi in reference.ROIS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index, image in enumerate(reference.canonical_image_ids()):
            writer.writerow([image] + [format(float(sources[(model, roi)].means[index]), ".17g")
                                      for model in models for roi in reference.ROIS])
    combined = reference.load_combined_means(path, models)
    for model in models:
        for roi in reference.ROIS:
            if not np.allclose(combined[f"{model}_{roi}"], sources[(model, roi)].means,
                               rtol=reference.MEAN_RELATIVE_TOLERANCE, atol=reference.MEAN_ABSOLUTE_TOLERANCE):
                raise reference.ReferenceValidationError(f"Refreshed means differ from raw predictions: {model}/{roi}")


def canonicalize_source(source: reference.ValidatedSource) -> reference.ValidatedSource:
    """Analyze precisely the FP32 predictions stored in NPZ/brain payloads.

    Decimal CSVs round-trip the float32 values, but their float64 interpretation
    is not those exact bits. Tiny-variance columns amplify that difference when
    standardized. Preserve the original source identity/hash and only select
    the explicit, versioned analysis representation.
    """
    voxels = source.voxels.astype(np.float32).astype(np.float64)
    if not np.isfinite(voxels).all():
        raise reference.ReferenceValidationError("Canonical FP32 predictions contain non-finite values")
    return replace(source, voxels=voxels, means=voxels.mean(axis=1, dtype=np.float64))


def cluster_refreshed_source(job: tuple[Path, Path]) -> tuple[dict, dict]:
    """Independent opt-in fits; algorithms and every fitting default stay unchanged."""
    source_path, data_root = job
    return (
        expansion.cluster_single_voxel_csv(source_path, data_root / expansion.CLUSTER_DIRS[0], canonical_float32=True),
        expansion.cluster_single_voxel_csv_hierarchical_corr(source_path, data_root / expansion.CLUSTER_DIRS[1], canonical_float32=True),
    )


def build_refreshed_reference(
    *, models: Sequence[str], raw_root: Path, stage_root: Path,
    baseline_root: Path = STANDARDIZATION_ROOT / "baseline",
    core_registry: Path = REPO_ROOT.parent / "cortex-core/cortex/models/__init__.py",
    app_constants: Path = REPO_ROOT.parent / "cortex-app/assets/js/constants.jsx",
    image_root: Path = REPO_ROOT / "public/images",
    parent_version: str = PARENT_REFERENCE_VERSION,
    parent_manifest_sha256: str = PARENT_MANIFEST_SHA256,
    generated_at: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise reference.ReferenceValidationError("Refresh clustering requires one to four isolated CPU workers")
    stage_root = stage_root.resolve()
    inputs = (baseline_root, raw_root, image_root)
    if stage_root.exists() or stage_root.is_symlink():
        raise reference.ReferenceValidationError("Refresh staging destination already exists; use a fresh path")
    if any(stage_root == root.resolve() or stage_root in root.resolve().parents
           or root.resolve() in stage_root.parents for root in inputs):
        raise reference.ReferenceValidationError("Refresh staging cannot overlap any input directory")
    parent, snapshot = validate_parent_snapshot(baseline_root, parent_version=parent_version,
                                               parent_manifest_sha256=parent_manifest_sha256)
    parent_artifacts = baseline_root / "artifacts/reference"
    parent_data = baseline_root / "public/data"
    target = validate_refresh_roster(models, parent, parent_data / "murty185_zero_index.json")
    expansion.validate_model_identifiers(target, core_registry=core_registry, app_constants=app_constants)
    model_directories = [path for path in raw_root.iterdir() if path.is_dir()]
    if any(path.is_symlink() for path in model_directories) or {path.name for path in model_directories} != set(target):
        raise reference.ReferenceValidationError("Fresh raw directory must contain exactly the full target model roster")
    _, parent_sources = expansion.load_baseline_sources(
        parent, artifact_root=parent_artifacts, raw_root=baseline_root / "data/image_by_model", data_root=parent_data,
    )
    sources, decisions = expansion.load_new_sources(raw_root, target, parent_sources)
    for source in sources.values():
        if f"data/image_by_model/{source.candidate.path.name}" in snapshot["files"]:
            raise reference.ReferenceValidationError("Fresh source filename collides with a retained historical prediction")
    proofs = validate_inference_proofs(models=target, raw_root=raw_root, sources=sources,
                                       core_root=core_registry.parents[2], image_root=image_root)
    sources = {key: canonicalize_source(source) for key, source in sources.items()}
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{stage_root.name}-refresh-", dir=stage_root.parent))
    try:
        artifact_root, data_root = work / "artifacts/reference", work / "public/data"
        shutil.copytree(parent_artifacts, artifact_root)
        shutil.copytree(parent_data, data_root)
        write_combined_means(data_root / "combined_image_means.csv", target, sources)
        summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in expansion.CLUSTER_DIRS}
        jobs = [(source.candidate.path, data_root) for _, source in sorted(sources.items())]
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            outputs = executor.map(cluster_refreshed_source, jobs) if executor else map(cluster_refreshed_source, jobs)
            for ((model, roi), source), raw_summaries in zip(sorted(sources.items()), outputs):
                print(f"Clustered refreshed reference {model}/{roi}", flush=True)
                for directory, summary in zip(expansion.CLUSTER_DIRS, raw_summaries):
                    normalized = expansion._normalized_summary(summary, directory)
                    summaries[directory].append(normalized)
                    expansion.write_json(data_root / directory / f"{source.candidate.path.stem}_summary.json", normalized)
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        for directory in expansion.CLUSTER_DIRS:
            merged = expansion.merge_cluster_summaries([], summaries[directory], target)
            expansion.write_json(data_root / directory / "voxel_cluster_summary.json", merged)
            pd.DataFrame(merged).to_csv(data_root / directory / "voxel_cluster_summary.csv", index=False)
        contracts = reference.load_static_voxel_cluster_contracts(
            sources, kmeans_dir=data_root / expansion.CLUSTER_DIRS[0], hierarchical_dir=data_root / expansion.CLUSTER_DIRS[1],
        )
        artifacts = {}
        for (model, roi), source in sorted(sources.items()):
            arrays, metadata = reference.fit_reference_arrays(source)
            contract = contracts[(model, roi)]
            if (not np.array_equal(arrays["kmeans_labels"], contract.kmeans_labels)
                    or not np.array_equal(arrays["hierarchical_labels"], contract.hierarchical_labels)):
                raise reference.ReferenceValidationError(f"Refreshed fitted labels disagree with static output: {model}/{roi}")
            arrays, alignment = reference.align_plot_projection_to_static(
                arrays, contract.plot_coordinates, context=f"{model}/{roi}", residual_tolerance=1e-9,
            )
            relative = Path("models") / model / f"{roi}.npz"
            path = artifact_root / relative
            reference.write_deterministic_npz(path, arrays)
            artifacts.setdefault(model, {})[roi] = expansion._new_artifact_entry(
                source, arrays, {**metadata, **alignment}, contract, path, relative,
            )
            print(f"Built refreshed artifact {model}/{roi}", flush=True)
        visual = parent["optional_artifacts"]["visual"]
        static_sha = reference.sha256_json({
            "voxel": {f"{model}/{roi}": {"kmeans": contract.kmeans_sha256, "hierarchical": contract.hierarchical_sha256}
                      for (model, roi), contract in sorted(contracts.items())},
            "visual": {"kmeans": visual["static_kmeans_sha256"], "hierarchical": visual["static_hierarchical_sha256"]},
        })
        combined_sha = digest(data_root / "combined_image_means.csv")
        identity = reference._reference_identity(
            models=target, image_ids=reference.canonical_image_ids(), combined_means_sha256=combined_sha,
            selected=sources, visual_features_sha256=visual["source_sha256"],
            visual_images_sha256=visual["source_images_sha256"], static_clusters_sha256=static_sha,
        )
        standardization = {
            "policy": POLICY, "parent_reference_version": parent["reference_version"],
            "analysis_input_policy": ANALYSIS_INPUT_POLICY,
            "parent_manifest_sha256": digest(parent_artifacts / "manifest.json"),
            "parent_snapshot_sha256": digest(baseline_root / "snapshot.json"),
            "parent_library_versions": parent["library_versions"], "regenerated_models": list(target),
            "inference_provenance_sha256": {model: digest(raw_root / model / "provenance.json") for model in target},
            "prediction_policies": {model: expansion.precision_policy_from_provenance(model, proofs[model]) for model in target},
            "model_prediction_policies": {model: expansion.prediction_policy_from_provenance(model, proofs[model]) for model in target},
        }
        identity["prediction_standardization"] = standardization
        version = reference.sha256_json(identity)
        manifest = copy.deepcopy(parent)
        manifest.pop("coverage_extension", None)
        manifest.update({
            "reference_version": version, "checksum": version, "artifacts": artifacts,
            "source_selection": decisions, "combined_means": {"file": "combined_image_means.csv", "sha256": combined_sha},
            "library_versions": identity["library_versions"], "prediction_standardization": standardization,
            "generator_git": reference._git_state(REPO_ROOT),
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        expansion.write_json(artifact_root / "manifest.json", manifest)
        expansion.write_json(data_root / "reference_manifest.json", reference.public_manifest_from_full(manifest))
        rewritten_private = ["manifest.json"] + [parent["artifacts"][model][roi]["path"] for model in target for roi in reference.ROIS]
        preserved_private = expansion.assert_preserved_files(parent_artifacts, artifact_root,
                                                             changed_relative_paths=rewritten_private)
        preserved_public = expansion.assert_preserved_files(parent_data, data_root,
                                                            changed_relative_paths=list(REWRITTEN_PUBLIC_PATHS))
        expansion.write_json(work / "preservation_report.json", {
            "policy": POLICY, "parent_reference_version": parent["reference_version"], "reference_version": version,
            "parent_snapshot_sha256": standardization["parent_snapshot_sha256"], "regenerated_models": list(target),
            "private_artifact_sha256": preserved_private, "public_artifact_sha256": preserved_public,
            "refitted_models": list(target), "live_install_performed": False,
        })
        if stage_root.exists():
            raise reference.ReferenceValidationError("Refresh staging destination appeared; no live data was installed")
        os.rename(work, stage_root)
        return manifest
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    roster = parser.add_mutually_exclusive_group(required=True)
    roster.add_argument("--models", nargs="+", help="Explicit full target roster, exactly matching the reviewed parent")
    roster.add_argument("--all-parent-models", action="store_true", help="Explicitly regenerate every model in the byte-verified parent snapshot")
    parser.add_argument("--raw-root", type=Path, default=STANDARDIZATION_ROOT / "raw-fp32")
    parser.add_argument("--stage-root", type=Path, default=STANDARDIZATION_ROOT / "staging")
    parser.add_argument("--baseline-root", type=Path, default=STANDARDIZATION_ROOT / "baseline")
    parser.add_argument("--workers", type=int, default=1, help="One to four isolated CPU clustering workers; fit defaults unchanged")
    args = parser.parse_args()
    models = document(child(args.baseline_root, "artifacts/reference/manifest.json"))["supported_models"] if args.all_parent_models else args.models
    manifest = build_refreshed_reference(models=models, raw_root=args.raw_root,
                                         stage_root=args.stage_root, baseline_root=args.baseline_root, workers=args.workers)
    print(f"Staged full {len(manifest['supported_models'])}-model refresh {manifest['reference_version']}; live data unchanged.")


if __name__ == "__main__":
    main()
