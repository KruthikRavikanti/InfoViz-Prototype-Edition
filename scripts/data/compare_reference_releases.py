#!/usr/bin/env python3
"""Quantify a prediction refresh against its preserved parent, without fitting.

This report measures numerical, rank, and cluster changes; it does not claim
that a small prediction change makes a model more biologically accurate.
Cluster comparisons use adjusted Rand indices, not arbitrary label numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
ROIS = ("ffa", "ppa", "eba")
CANONICAL_IMAGES = [f"image_{index:03d}.png" for index in range(1, 186)]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def safe_child(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Unsafe release artifact path")
    return path


def compare_arrays(before: np.ndarray, after: np.ndarray, *,
                   before_means: np.ndarray | None = None,
                   after_means: np.ndarray | None = None) -> dict:
    """Compare raw responses and the means actually used for image rankings."""
    before = np.asarray(before)
    after = np.asarray(after)
    if (before.shape != after.shape or before.ndim != 2 or not before.size
            or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError("Comparable finite, equally shaped raw response matrices are required")
    delta = after.astype(np.float64) - before.astype(np.float64)
    before_raw_means = before.mean(axis=1, dtype=np.float64)
    after_raw_means = after.mean(axis=1, dtype=np.float64)
    if (before_means is None) != (after_means is None):
        raise ValueError("Both releases' saved ranking means must be supplied together")
    if before_means is None:
        before_means, after_means = before_raw_means, after_raw_means
    else:
        before_means = np.asarray(before_means, dtype=np.float64)
        after_means = np.asarray(after_means, dtype=np.float64)
        if (before_means.shape != (before.shape[0],) or after_means.shape != before_means.shape
                or not np.isfinite(before_means).all() or not np.isfinite(after_means).all()):
            raise ValueError("Saved ranking means must be finite and match image identity/order")
    before_ranks = rankdata(-before_means, method="average")
    after_ranks = rankdata(-after_means, method="average")
    if before_ranks.std() == 0 or after_ranks.std() == 0:
        rank_correlation = None  # Undefined correlation, not a misleading NaN.
    elif np.array_equal(before_ranks, after_ranks):
        rank_correlation = 1.0
    else:
        rank_correlation = float(np.corrcoef(before_ranks, after_ranks)[0, 1])
    return {
        "maximum_absolute_voxel_change": float(np.abs(delta).max()),
        "mean_absolute_voxel_change": float(np.abs(delta).mean()),
        "maximum_absolute_raw_voxel_mean_change": float(np.abs(after_raw_means - before_raw_means).max()),
        "maximum_absolute_mean_response_change": float(np.abs(after_means - before_means).max()),
        "mean_response_rank_spearman": rank_correlation,
        "images_with_changed_mean_response_midrank": int(np.count_nonzero(before_ranks != after_ranks)),
        "maximum_mean_response_midrank_change": float(np.abs(after_ranks - before_ranks).max()),
    }


def compare_atlas(before: Path, after: Path) -> float:
    before_frame = pd.read_csv(before)
    after_frame = pd.read_csv(after)
    columns = ["voxel_index", "roi", "roi_voxel_index"]
    if (len(before_frame) != 844 or not before_frame[columns].equals(after_frame[columns])
            or before_frame["cluster_label"].isna().any() or after_frame["cluster_label"].isna().any()):
        raise ValueError("Selectivity atlas voxel identity/order changed")
    return float(adjusted_rand_score(before_frame["cluster_label"], after_frame["cluster_label"]))


def compare_releases(before_root: Path, after_root: Path, spatial_root: Path,
                     *, before_raw: Path | None = None, after_raw: Path | None = None) -> dict:
    before_artifacts = before_root / "artifacts/reference"
    after_artifacts = after_root / "artifacts/reference"
    before = document(before_artifacts / "manifest.json")
    after = document(after_artifacts / "manifest.json")
    models = before["supported_models"]
    if (models != after["supported_models"] or before["rois"] != list(ROIS)
            or after["rois"] != list(ROIS) or before["reference_count"] != 185
            or after["reference_count"] != 185):
        raise ValueError("Refresh changed the model, region, or reference roster")
    if digest(before_root / "public/data/murty185_zero_index.json") != digest(after_root / "public/data/murty185_zero_index.json"):
        raise ValueError("Historical scoreboard scores unexpectedly changed")
    legacy = set(before.get("coverage_extension", {}).get("baseline_models", []))
    policies = after["prediction_standardization"]["model_prediction_policies"]
    records = []
    state_checks = []
    for model in models:
        if before_raw is not None and after_raw is not None and (before_raw / model / "provenance.json").is_file():
            old_proof = document(before_raw / model / "provenance.json")
            new_proof = document(after_raw / model / "provenance.json")
            same_state = old_proof["backbone_state_sha256"] == new_proof["backbone_state_sha256"]
            same_readouts = all(old_proof["rois"][roi]["checkpoint_sha256"] == new_proof["rois"][roi]["checkpoint_sha256"] for roi in ROIS)
            if not same_state or not same_readouts:
                raise ValueError(f"Previously FP32 model weights/readouts changed: {model}")
            state_checks.append(model)
        for roi in ROIS:
            old_entry = before["artifacts"][model][roi]
            new_entry = after["artifacts"][model][roi]
            old_path = safe_child(before_artifacts, old_entry["path"])
            new_path = safe_child(after_artifacts, new_entry["path"])
            if digest(old_path) != old_entry["sha256"] or digest(new_path) != new_entry["sha256"]:
                raise ValueError(f"Private artifact checksum failed: {model}/{roi}")
            with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
                if old["image_ids"].tolist() != CANONICAL_IMAGES or new["image_ids"].tolist() != CANONICAL_IMAGES:
                    raise ValueError(f"Image identity/order changed: {model}/{roi}")
                metrics = compare_arrays(old["reference_voxels"], new["reference_voxels"],
                                         before_means=old["reference_means"], after_means=new["reference_means"])
                metrics.update({
                    "kmeans_cluster_adjusted_rand": float(adjusted_rand_score(old["kmeans_labels"], new["kmeans_labels"])),
                    "hierarchical_cluster_adjusted_rand": float(adjusted_rand_score(old["hierarchical_labels"], new["hierarchical_labels"])),
                })
            records.append({"model": model, "roi": roi,
                            "previous_policy_group": "original_26_legacy" if model in legacy else "previously_fp32",
                            "stochastic": policies[model]["response_policy"] == "seeded_stochastic_sampling",
                            **metrics})
    atlases = [{"model": model, "cluster_adjusted_rand": compare_atlas(
        before_root / "public/data/voxel_selectivity_atlas" / model / "points.csv",
        spatial_root / "public/data/voxel_selectivity_atlas" / model / "points.csv")}
        for model in models]
    return {"before_reference_version": before["reference_version"], "after_reference_version": after["reference_version"],
            "model_count": len(models), "roi_count": len(ROIS), "reference_count": 185,
            "analysis_input_policy": after["prediction_standardization"]["analysis_input_policy"],
            "ranking_mean_source": "saved_npz_reference_means_used_by_the_application",
            "scoreboard_bytes_unchanged": True, "previously_fp32_weight_identities_verified": state_checks,
            "response_and_image_cluster_changes": records, "selectivity_cluster_changes": atlases,
            "interpretation": "These are computational changes, not evidence of improved biological prediction accuracy. Original-26 archives lack backbone-state provenance, so differences cannot all be attributed solely to arithmetic precision; AlexNetRandom initialization is now explicitly seed-controlled. Current AlexNetRandom reference/live consistency does not prove its historical training-encoder pairing, whose initialization is undocumented. Previously FP32 backbone/readout identities are checked separately. Stochastic-model changes compare saved samples, not expected responses. ARI handles cluster-label permutations; similarity-space coordinates are not anatomical."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-spatial-root", type=Path, required=True)
    parser.add_argument("--before-raw-root", type=Path)
    parser.add_argument("--after-raw-root", type=Path)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    allowed = (REPO_ROOT / "artifacts/precision-standardization").resolve()
    target = args.report_path.resolve()
    if not target.is_relative_to(allowed) or target == allowed or args.report_path.exists():
        parser.error("Report must be a new file inside artifacts/precision-standardization")
    report = compare_releases(args.before_root, args.after_root, args.after_spatial_root,
                              before_raw=args.before_raw_root, after_raw=args.after_raw_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"Compared {report['model_count']} models across {report['roi_count']} ROIs; saved {args.report_path}")


if __name__ == "__main__":
    main()
