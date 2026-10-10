#!/usr/bin/env python3
"""Stage an explicit append-only expansion of the audited Murty-185 bundle.

This command never installs live data. Existing prediction/cluster artifacts are
copied unchanged; only explicitly requested new, complete NSD-1000 model triads
are fitted. New inference is a separate prerequisite, not part of this builder.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from . import build_reference_artifacts as reference
    from .cluster_runner import cluster_single_voxel_csv, cluster_single_voxel_csv_hierarchical_corr
    from .validate_model_registries import validate_model_identifiers
except ImportError:  # Direct CLI invocation.
    import build_reference_artifacts as reference  # type: ignore
    from cluster_runner import cluster_single_voxel_csv, cluster_single_voxel_csv_hierarchical_corr  # type: ignore
    from validate_model_registries import validate_model_identifiers  # type: ignore


REPO_ROOT = reference.REPO_ROOT
POLICY = "append_only_preserve_baseline_artifacts_v1"
CLUSTER_DIRS = ("voxel_clusters", "voxel_clusters_hierarchical_corr")
# The registered RN50 readout uses the unchanged resnet50_ns constructor, whose
# actual VOne block has no neuronal noise. Do not infer policy from its name.
STOCHASTIC_VONE_MODELS = frozenset({"vone_alexnet", "vone_cornet_s"})
FP32_NUMERIC_POLICY = {
    "policy": "fp32_no_tf32_v1",
    "output_dtype": "float32",
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_autocast_enabled": False,
    "cpu_autocast_enabled": False,
}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise reference.ReferenceValidationError(f"Cannot read JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reference._json_text(value), encoding="utf-8")


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise reference.ReferenceValidationError(f"Invalid artifact path inside {root}: {relative}")
    return path


def validate_baseline_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != reference.DATASET
        or manifest.get("reference_count") != reference.REFERENCE_COUNT
        or manifest.get("image_count") != reference.REFERENCE_COUNT
        or manifest.get("rois") != list(reference.ROIS)
        or manifest.get("voxel_counts") != reference.VOXEL_COUNTS
        or manifest.get("image_ids_sha256") != reference.sha256_json(list(reference.canonical_image_ids()))
        or not isinstance(manifest.get("reference_version"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", manifest["reference_version"])
        or manifest.get("checksum") != manifest.get("reference_version")
        or manifest.get("clustering") != reference._clustering_manifest()
        or manifest.get("artifact_schema") != reference._artifact_schema_manifest()
    ):
        raise reference.ReferenceValidationError("Baseline reference identity/dimensions/pipeline are incompatible")
    models = manifest.get("supported_models")
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not re.fullmatch(r"[a-z0-9_]+(?:\.[0-9]+)?", model) for model in models)
        or len(models) != len(set(models))
        or models != sorted(models)
        or set(manifest.get("artifacts", {})) != set(models)
    ):
        raise reference.ReferenceValidationError("Baseline model roster/artifact coverage is invalid")
    for model in models:
        if set(manifest["artifacts"][model]) != set(reference.ROIS):
            raise reference.ReferenceValidationError(f"Baseline model {model} has an incomplete ROI triad")
    return tuple(models)


def validate_target_roster(models: Sequence[str], baseline_models: Sequence[str]) -> tuple[str, ...]:
    if (
        not models
        or len(models) != len(set(models))
        or any(not isinstance(model, str) or not re.fullmatch(r"[a-z0-9_]+(?:\.[0-9]+)?", model) for model in models)
    ):
        raise reference.ReferenceValidationError("Target roster must contain unique safe exact model identifiers")
    removed = sorted(set(baseline_models) - set(models))
    if removed:
        raise reference.ReferenceValidationError(f"Append-only expansion cannot remove baseline models: {removed}")
    if set(models) == set(baseline_models):
        raise reference.ReferenceValidationError("Target roster does not add any new models")
    return tuple(sorted(models))


def validate_scoreboard_roster(models: Sequence[str], scoreboard_path: Path) -> None:
    """Coverage may use real scored models, never the non-inferable ceiling row."""
    scoreboard = read_json(scoreboard_path)
    if not isinstance(scoreboard, dict) or any(not isinstance(scoreboard.get(roi), dict) for roi in reference.ROIS):
        raise reference.ReferenceValidationError("Scoreboard must have FFA/PPA/EBA model score dictionaries")
    rosters = [set(scoreboard[roi]) - {"ceiling"} for roi in reference.ROIS]
    if any(roster != rosters[0] for roster in rosters[1:]):
        raise reference.ReferenceValidationError("Scoreboard model coverage differs between ROIs")
    unexpected = sorted(set(models) - rosters[0])
    if unexpected:
        raise reference.ReferenceValidationError(f"Requested coverage includes non-model/unscored identifiers: {unexpected}")


def prediction_policy_from_provenance(model: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Keep VOne neuronal noise unchanged and label saved draws, not averages."""
    expected = {
        "response_policy": "seeded_stochastic_sampling" if model in STOCHASTIC_VONE_MODELS else "evaluation_mode",
        "deterministic_per_stimulus": model not in STOCHASTIC_VONE_MODELS,
        "noise_modes": ["neuronal"] if model in STOCHASTIC_VONE_MODELS else [],
    }
    for key, value in expected.items():
        if provenance.get(key) != value or (key == "deterministic_per_stimulus" and not isinstance(provenance.get(key), bool)):
            raise reference.ReferenceValidationError(f"New {model} response/noise policy disagrees with its unmodified constructor: {key}")
    return expected


def precision_policy_from_provenance(model: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Require observed FP32 flags, not an output cast or an assumed dtype.

    TF32 inference can return float32 arrays while still using reduced-precision
    multiplication. Every added model must therefore carry the complete locked
    runtime policy and the identity of the helper that applied it.
    """
    numeric = provenance.get("numeric_policy")
    if not isinstance(numeric, dict) or set(numeric) != set(FP32_NUMERIC_POLICY):
        raise reference.ReferenceValidationError(f"New {model} inference lacks the complete explicit FP32 numeric policy")
    for key, expected in FP32_NUMERIC_POLICY.items():
        actual = numeric[key]
        if type(actual) is not type(expected) or actual != expected:
            raise reference.ReferenceValidationError(f"New {model} inference does not use the audited FP32 numeric policy: {key}")
    helper_sha256 = provenance.get("precision_helper_sha256")
    if not isinstance(helper_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", helper_sha256):
        raise reference.ReferenceValidationError(f"New {model} inference lacks the precision helper source identity")
    source_hashes = provenance.get("core_inference_source_sha256")
    if not isinstance(source_hashes, dict) or source_hashes.get("cortex/utils/inference_precision.py") != helper_sha256:
        raise reference.ReferenceValidationError(f"New {model} precision helper identity disagrees with its Core source lineage")
    return {"numeric_policy": dict(numeric), "precision_helper_sha256": helper_sha256}


def _validate_baseline_arrays(path: Path, entry: Mapping[str, Any], source: reference.ValidatedSource) -> None:
    if reference.sha256_file(path) != entry["sha256"]:
        raise reference.ReferenceValidationError(f"Baseline NPZ checksum changed: {path}")
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(reference.ARRAY_CONTRACT):
                raise reference.ReferenceValidationError(f"Baseline NPZ array contract changed: {path}")
            for name in arrays.files:
                array = arrays[name]
                expected = entry["arrays"][name]
                if str(array.dtype) != expected["dtype"] or list(array.shape) != expected["shape"]:
                    raise reference.ReferenceValidationError(f"Baseline NPZ {name} metadata mismatch: {path}")
                if array.dtype.hasobject or (np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
                    raise reference.ReferenceValidationError(f"Baseline NPZ {name} values are invalid: {path}")
            if arrays["image_ids"].tolist() != list(source.image_ids):
                raise reference.ReferenceValidationError(f"Baseline NPZ reference image order changed: {path}")
            if not np.array_equal(arrays["reference_voxels"], source.voxels.astype(np.float32)):
                raise reference.ReferenceValidationError(f"Baseline NPZ is not its raw response matrix: {path}")
            if not np.array_equal(arrays["reference_means"], source.means):
                raise reference.ReferenceValidationError(f"Baseline NPZ means are not raw float64 means: {path}")
    except (OSError, ValueError, KeyError) as error:
        raise reference.ReferenceValidationError(f"Cannot validate baseline NPZ {path}: {error}") from error


def load_baseline_sources(
    manifest: Mapping[str, Any], *, artifact_root: Path, raw_root: Path, data_root: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], reference.ValidatedSource]]:
    models = validate_baseline_manifest(manifest)
    combined_path = data_root / "combined_image_means.csv"
    if reference.sha256_file(combined_path) != manifest["combined_means"]["sha256"]:
        raise reference.ReferenceValidationError("Baseline aggregate means checksum changed")
    combined = reference.load_combined_means(combined_path, models)
    public_manifest = read_json(data_root / "reference_manifest.json")
    if public_manifest != reference.public_manifest_from_full(manifest):
        raise reference.ReferenceValidationError("Public and private baseline reference manifests disagree")
    selected: dict[tuple[str, str], reference.ValidatedSource] = {}
    for model in models:
        for roi in reference.ROIS:
            entry = manifest["artifacts"][model][roi]
            source_name = entry["source_file"]
            if Path(source_name).name != source_name:
                raise reference.ReferenceValidationError("Baseline source filename must be a basename")
            path = checked_path(raw_root, source_name)
            match = reference.FILENAME_RE.fullmatch(source_name)
            if not match or match.group("model") != model or match.group("roi") != roi:
                raise reference.ReferenceValidationError(f"Baseline source filename identity disagrees for {model}/{roi}")
            candidate = reference.SourceCandidate(model, roi, match.group("timestamp"), path)
            source = reference.load_and_validate_source(
                candidate, expected_ids=reference.canonical_image_ids(),
                expected_voxel_count=reference.VOXEL_COUNTS[roi],
                expected_means=combined[f"{model}_{roi}"].to_numpy(dtype=np.float64),
            )
            if source.sha256 != entry["source_sha256"]:
                raise reference.ReferenceValidationError(f"Baseline raw source checksum changed: {model}/{roi}")
            selection = manifest["source_selection"][f"{model}/{roi}"]
            if selection["selected"] != source_name:
                raise reference.ReferenceValidationError(f"Baseline source selection disagrees: {model}/{roi}")
            _validate_baseline_arrays(checked_path(artifact_root, entry["path"]), entry, source)
            selected[(model, roi)] = source
    contracts = reference.load_static_voxel_cluster_contracts(
        selected, kmeans_dir=data_root / CLUSTER_DIRS[0], hierarchical_dir=data_root / CLUSTER_DIRS[1],
    )
    for (model, roi), contract in contracts.items():
        entry = manifest["artifacts"][model][roi]
        if (contract.kmeans_sha256 != entry["static_kmeans_sha256"]
                or contract.hierarchical_sha256 != entry["static_hierarchical_sha256"]):
            raise reference.ReferenceValidationError(f"Baseline static cluster checksum changed: {model}/{roi}")
        with np.load(artifact_root / entry["path"], allow_pickle=False) as arrays:
            if (not np.array_equal(arrays["plot_reference_coordinates"], contract.plot_coordinates)
                    or not np.array_equal(arrays["kmeans_labels"], contract.kmeans_labels)
                    or not np.array_equal(arrays["hierarchical_labels"], contract.hierarchical_labels)):
                raise reference.ReferenceValidationError(f"Baseline static coordinates/labels disagree: {model}/{roi}")
    for name, entry in manifest["optional_artifacts"].items():
        if entry.get("available", True) and "path" in entry:
            if reference.sha256_file(checked_path(artifact_root, entry["path"])) != entry["sha256"]:
                raise reference.ReferenceValidationError(f"Baseline optional artifact checksum changed: {name}")
    return combined, selected


def load_new_sources(
    new_raw_root: Path, models: Sequence[str], baseline_sources: Mapping[tuple[str, str], reference.ValidatedSource],
) -> tuple[dict[tuple[str, str], reference.ValidatedSource], dict[str, Any]]:
    """Read atomic per-model inference folders; never silently select a partial triad."""
    selected: dict[tuple[str, str], reference.ValidatedSource] = {}
    decisions: dict[str, Any] = {}
    templates = {roi: next(source for (_, candidate_roi), source in baseline_sources.items() if candidate_roi == roi)
                 for roi in reference.ROIS}
    for model in models:
        model_dir = new_raw_root / model
        candidates = reference.discover_source_candidates(model_dir, expected_models=[model])
        provenance_path = model_dir / "provenance.json"
        provenance = read_json(provenance_path)
        if not isinstance(provenance, dict) or provenance.get("model") != model or provenance.get("dataset") != reference.DATASET:
            raise reference.ReferenceValidationError(f"New inference provenance identity disagrees for {model}")
        prediction_policy_from_provenance(model, provenance)
        precision_policy_from_provenance(model, provenance)
        for roi in reference.ROIS:
            choices = candidates[(model, roi)]
            if len(choices) != 1:
                raise reference.ReferenceValidationError(f"New inference {model}/{roi} must contain exactly one audited CSV")
            candidate = choices[0]
            try:
                frame = pd.read_csv(candidate.path)
                id_column = reference._strict_id_column(frame.columns, candidate.path)
                columns = tuple(str(column) for column in frame.columns if column != id_column)
                if columns != templates[roi].voxel_columns:
                    raise reference.ReferenceValidationError(f"New {model}/{roi} voxel identity/order differs from the audited baseline")
                numeric = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
                expected_means = numeric.mean(axis=1, dtype=np.float64)
            except (ValueError, TypeError, KeyError) as error:
                raise reference.ReferenceValidationError(f"New {model}/{roi} CSV is invalid: {error}") from error
            source = reference.load_and_validate_source(
                candidate, expected_ids=reference.canonical_image_ids(),
                expected_voxel_count=reference.VOXEL_COUNTS[roi], expected_means=expected_means,
            )
            raw_roi = provenance.get("rois", {}).get(roi, {})
            # The pilot generator used path/sha256; later records add explicit aliases.
            # Both representations must resolve to this exact source if both exist.
            source_name = raw_roi.get("source_file")
            source_hash = raw_roi.get("source_sha256", raw_roi.get("sha256"))
            stored_path = raw_roi.get("path")
            if source_name is None and isinstance(stored_path, str):
                source_name = Path(stored_path).name
            if (source_name != candidate.path.name or source_hash != source.sha256
                    or ("sha256" in raw_roi and raw_roi["sha256"] != source.sha256)
                    or (stored_path is not None and (not isinstance(stored_path, str)
                        or (model_dir / stored_path).resolve() != candidate.path.resolve()))):
                raise reference.ReferenceValidationError(f"New inference provenance does not verify the source for {model}/{roi}")
            selected[(model, roi)] = source
            decisions[f"{model}/{roi}"] = {
                "selected": candidate.path.name, "selected_timestamp": candidate.timestamp,
                "candidate_count": 1, "rejected_newer_candidates": [],
                "inference_provenance_sha256": reference.sha256_file(provenance_path),
                "raw_relative_path": f"{model}/{candidate.path.name}",
            }
    return selected, decisions


def append_combined_means(
    baseline_path: Path, output_path: Path, models: Sequence[str],
    sources: Mapping[tuple[str, str], reference.ValidatedSource],
) -> None:
    """Retain every existing CSV value string, adding only true raw float64 means."""
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_columns = list(reader.fieldnames or [])
        rows = list(reader)
    columns = ["image_name"] + [f"{model}_{roi}" for model in sorted(models) for roi in reference.ROIS]
    if not set(old_columns).issubset(columns):
        raise reference.ReferenceValidationError("Expanded aggregate cannot drop existing columns")
    if [row["image_name"] for row in rows] != list(reference.canonical_image_ids()):
        raise reference.ReferenceValidationError("Baseline aggregate canonical image order changed")
    for row_index, row in enumerate(rows):
        for (model, roi), source in sources.items():
            column = f"{model}_{roi}"
            if column not in row:
                row[column] = format(float(source.means[row_index]), ".17g")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    expanded = reference.load_combined_means(output_path, models)
    old_models = sorted({model for model, _ in sources if f"{model}_ffa" in old_columns})
    old_frame = reference.load_combined_means(baseline_path, old_models)
    for column in old_columns[1:]:
        if not np.array_equal(old_frame[column].to_numpy(), expanded[column].to_numpy()):
            raise reference.ReferenceValidationError(f"Expanded aggregate altered baseline values: {column}")


def merge_cluster_summaries(
    baseline: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]], models: Sequence[str],
) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(dict(row)) for row in baseline] + [copy.deepcopy(dict(row)) for row in additions]
    keys = [(row.get("model"), row.get("roi")) for row in rows]
    expected = {(model, roi) for model in models for roi in reference.ROIS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise reference.ReferenceValidationError("Expanded static cluster summary requires every model/ROI exactly once")
    if any(row.get("n_images") != reference.REFERENCE_COUNT
           or row.get("n_voxels") != reference.VOXEL_COUNTS[row["roi"]] for row in rows):
        raise reference.ReferenceValidationError("Expanded static cluster summary dimensions are invalid")
    return rows


def _normalized_summary(summary: Mapping[str, Any], directory: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(summary))
    for name in ("result_csv", "linkage_csv", "dendrogram_png"):
        if result.get(name):
            result[name] = f"public/data/{directory}/{Path(result[name]).name}"
    return result


def assert_preserved_files(
    baseline_root: Path, copied_root: Path, *, changed_relative_paths: Sequence[str] = (),
) -> dict[str, str]:
    excluded = set(changed_relative_paths)
    hashes: dict[str, str] = {}
    for path in sorted(baseline_root.rglob("*")):
        if path.is_symlink():
            raise reference.ReferenceValidationError(f"Baseline bundles cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(baseline_root).as_posix()
        if relative in excluded:
            continue
        digest = reference.sha256_file(path)
        if not (copied_root / relative).is_file() or reference.sha256_file(copied_root / relative) != digest:
            raise reference.ReferenceValidationError(f"Append-only expansion changed baseline artifact bytes: {relative}")
        hashes[relative] = digest
    return hashes


def _new_artifact_entry(source: reference.ValidatedSource, arrays: Mapping[str, np.ndarray],
                        metadata: Mapping[str, Any], contract: reference.StaticClusterContract,
                        artifact_path: Path, relative_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(), "sha256": reference.sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
        "source_file": source.candidate.path.name, "source_timestamp": source.candidate.timestamp,
        "source_sha256": source.sha256, "static_kmeans_sha256": contract.kmeans_sha256,
        "static_hierarchical_sha256": contract.hierarchical_sha256,
        "image_count": len(source.image_ids), "voxel_count": source.voxels.shape[1],
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                   for name, array in sorted(arrays.items())}, **metadata,
    }


def build_expanded_reference(
    *, models: Sequence[str], new_raw_root: Path, stage_root: Path,
    baseline_artifact_root: Path = REPO_ROOT / "artifacts/reference",
    baseline_data_root: Path = REPO_ROOT / "public/data",
    baseline_raw_root: Path = REPO_ROOT / "data/image_by_model",
    core_registry: Path = REPO_ROOT.parent / "cortex-core/cortex/models/__init__.py",
    app_constants: Path = REPO_ROOT.parent / "cortex-app/assets/js/constants.jsx",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Validate and publish a new staging directory; live inputs are never written."""
    stage_root = stage_root.resolve()
    roots = (baseline_artifact_root, baseline_data_root, baseline_raw_root, new_raw_root)
    if stage_root.exists():
        raise reference.ReferenceValidationError(f"Staging destination already exists; use a fresh path: {stage_root}")
    if any(stage_root == root.resolve() or stage_root in root.resolve().parents
           or root.resolve() in stage_root.parents for root in roots):
        raise reference.ReferenceValidationError("Staging destination must not overlap any input directory")
    baseline_path = baseline_artifact_root / "manifest.json"
    baseline = read_json(baseline_path)
    baseline_models = validate_baseline_manifest(baseline)
    target = validate_target_roster(models, baseline_models)
    validate_scoreboard_roster(target, baseline_data_root / "murty185_zero_index.json")
    validate_model_identifiers(target, core_registry=core_registry, app_constants=app_constants)
    added_models = tuple(model for model in target if model not in baseline_models)
    _, baseline_sources = load_baseline_sources(
        baseline, artifact_root=baseline_artifact_root, raw_root=baseline_raw_root, data_root=baseline_data_root,
    )
    new_sources, decisions = load_new_sources(new_raw_root, added_models, baseline_sources)
    sources = {**baseline_sources, **new_sources}
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{stage_root.name}-build-", dir=stage_root.parent))
    try:
        artifact_root = work / "artifacts/reference"
        data_root = work / "public/data"
        shutil.copytree(baseline_artifact_root, artifact_root)
        shutil.copytree(baseline_data_root, data_root)
        append_combined_means(baseline_data_root / "combined_image_means.csv",
                              data_root / "combined_image_means.csv", target, sources)
        summary_additions: dict[str, list[dict[str, Any]]] = {name: [] for name in CLUSTER_DIRS}
        for (model, roi), source in sorted(new_sources.items()):
            print(f"Clustering new reference {model}/{roi}", flush=True)
            raw_summaries = (
                cluster_single_voxel_csv(source.candidate.path, data_root / CLUSTER_DIRS[0]),
                cluster_single_voxel_csv_hierarchical_corr(source.candidate.path, data_root / CLUSTER_DIRS[1]),
            )
            for directory, summary in zip(CLUSTER_DIRS, raw_summaries):
                normalized = _normalized_summary(summary, directory)
                summary_additions[directory].append(normalized)
                write_json(data_root / directory / f"{source.candidate.path.stem}_summary.json", normalized)
        for directory in CLUSTER_DIRS:
            baseline_summary = read_json(baseline_data_root / directory / "voxel_cluster_summary.json")
            merged = merge_cluster_summaries(baseline_summary, summary_additions[directory], target)
            write_json(data_root / directory / "voxel_cluster_summary.json", merged)
            pd.DataFrame(merged).to_csv(data_root / directory / "voxel_cluster_summary.csv", index=False)
        contracts = reference.load_static_voxel_cluster_contracts(
            sources, kmeans_dir=data_root / CLUSTER_DIRS[0], hierarchical_dir=data_root / CLUSTER_DIRS[1],
        )
        artifacts = copy.deepcopy(baseline["artifacts"])
        for (model, roi), source in sorted(new_sources.items()):
            arrays, metadata = reference.fit_reference_arrays(source)
            contract = contracts[(model, roi)]
            if (not np.array_equal(arrays["kmeans_labels"], contract.kmeans_labels)
                    or not np.array_equal(arrays["hierarchical_labels"], contract.hierarchical_labels)):
                raise reference.ReferenceValidationError(f"New fitted cluster labels disagree with static output: {model}/{roi}")
            arrays, alignment = reference.align_plot_projection_to_static(
                arrays, contract.plot_coordinates, context=f"{model}/{roi}", residual_tolerance=1e-9,
            )
            relative = Path("models") / model / f"{roi}.npz"
            path = artifact_root / relative
            reference.write_deterministic_npz(path, arrays)
            artifacts.setdefault(model, {})[roi] = _new_artifact_entry(
                source, arrays, {**metadata, **alignment}, contract, path, relative,
            )
            print(f"Built new artifact {model}/{roi}", flush=True)
        old_visual = baseline["optional_artifacts"]["visual"]
        static_hash = reference.sha256_json({
            "voxel": {f"{model}/{roi}": {"kmeans": contract.kmeans_sha256,
                                         "hierarchical": contract.hierarchical_sha256}
                      for (model, roi), contract in sorted(contracts.items())},
            "visual": {"kmeans": old_visual["static_kmeans_sha256"],
                       "hierarchical": old_visual["static_hierarchical_sha256"]},
        })
        combined_sha = reference.sha256_file(data_root / "combined_image_means.csv")
        identity = reference._reference_identity(
            models=target, image_ids=reference.canonical_image_ids(), combined_means_sha256=combined_sha,
            selected=sources, visual_features_sha256=old_visual["source_sha256"],
            visual_images_sha256=old_visual["source_images_sha256"], static_clusters_sha256=static_hash,
        )
        extension = {
            "policy": POLICY, "parent_reference_version": baseline["reference_version"],
            "baseline_manifest_sha256": reference.sha256_file(baseline_path),
            "baseline_models": list(baseline_models), "added_models": list(added_models),
            "baseline_library_versions": baseline["library_versions"],
            "new_inference_provenance_sha256": {
                model: reference.sha256_file(new_raw_root / model / "provenance.json") for model in added_models
            },
            "model_prediction_policies": {
                model: prediction_policy_from_provenance(model, read_json(new_raw_root / model / "provenance.json"))
                for model in added_models
            },
            "prediction_policies": {
                model: precision_policy_from_provenance(model, read_json(new_raw_root / model / "provenance.json"))
                for model in added_models
            },
        }
        identity["coverage_extension"] = extension
        version = reference.sha256_json(identity)
        manifest = copy.deepcopy(baseline)
        manifest.update({
            "reference_version": version, "checksum": version, "supported_models": list(target),
            "artifacts": artifacts, "source_selection": {**baseline["source_selection"], **decisions},
            "combined_means": {"file": "combined_image_means.csv", "sha256": combined_sha},
            "library_versions": identity["library_versions"], "coverage_extension": extension,
            "generator_git": reference._git_state(REPO_ROOT),
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        write_json(artifact_root / "manifest.json", manifest)
        write_json(data_root / "reference_manifest.json", reference.public_manifest_from_full(manifest))
        private_hashes = assert_preserved_files(baseline_artifact_root, artifact_root,
                                               changed_relative_paths=["manifest.json"])
        public_hashes = assert_preserved_files(
            baseline_data_root, data_root,
            changed_relative_paths=["combined_image_means.csv", "reference_manifest.json"]
                + [f"{directory}/voxel_cluster_summary.{suffix}" for directory in CLUSTER_DIRS for suffix in ("csv", "json")],
        )
        for model in baseline_models:
            if manifest["artifacts"][model] != baseline["artifacts"][model]:
                raise reference.ReferenceValidationError(f"Baseline artifact metadata changed for {model}")
        write_json(work / "preservation_report.json", {
            "policy": POLICY, "parent_reference_version": baseline["reference_version"], "reference_version": version,
            "baseline_models": list(baseline_models), "added_models": list(added_models),
            "private_artifact_sha256": private_hashes, "public_artifact_sha256": public_hashes,
            "live_install_performed": False,
        })
        # The destination is intentionally new. Never replace a user's existing staging tree.
        if stage_root.exists():
            raise reference.ReferenceValidationError("Staging destination appeared during generation; nothing was installed")
        os.rename(work, stage_root)
        return manifest
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Full explicit target roster including every baseline model")
    parser.add_argument("--new-raw-dir", type=Path, default=REPO_ROOT / "artifacts/coverage-expansion/raw")
    parser.add_argument("--stage-root", type=Path, default=REPO_ROOT / "artifacts/coverage-expansion/staging")
    parser.add_argument("--baseline-artifact-root", type=Path, default=REPO_ROOT / "artifacts/reference")
    parser.add_argument("--baseline-data-root", type=Path, default=REPO_ROOT / "public/data")
    parser.add_argument("--baseline-raw-root", type=Path, default=REPO_ROOT / "data/image_by_model")
    parser.add_argument("--core-registry", type=Path, default=REPO_ROOT.parent / "cortex-core/cortex/models/__init__.py")
    parser.add_argument("--app-constants", type=Path, default=REPO_ROOT.parent / "cortex-app/assets/js/constants.jsx")
    parser.add_argument("--generated-at", help="Optional fixed timestamp for reproducibility checks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = build_expanded_reference(
            models=args.models, new_raw_root=args.new_raw_dir, stage_root=args.stage_root,
            baseline_artifact_root=args.baseline_artifact_root, baseline_data_root=args.baseline_data_root,
            baseline_raw_root=args.baseline_raw_root, core_registry=args.core_registry, app_constants=args.app_constants,
            generated_at=args.generated_at,
        )
    except (reference.ReferenceValidationError, ValueError) as error:
        raise SystemExit(f"Reference coverage expansion failed: {error}") from error
    print(f"Staged {len(manifest['supported_models'])} models at {args.stage_root}; live data unchanged.")
    print(f"reference_version={manifest['reference_version']}")


if __name__ == "__main__":
    main()
