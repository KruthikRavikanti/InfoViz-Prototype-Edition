"""Build deterministic NSD-1000 reference artifacts for Cortex comparison sessions.

The raw voxel CSVs are authoritative, server-side inputs.  This builder validates
the complete 26 model x 3 ROI source set before writing anything, reproduces the
existing clustering pipelines from :mod:`cluster_runner`, and emits portable NPZ
files that can transform query stimuli without refitting reference clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from PIL import __version__ as pillow_version
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:  # Supports both direct CLI execution and package-style test imports.
    from .cluster_runner import (
        FILENAME_RE,
        choose_best_hierarchical_k,
        choose_best_k,
        clean_correlation_distance_vector,
        extract_visual_feature_frame,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    from cluster_runner import (  # type: ignore
        FILENAME_RE,
        choose_best_hierarchical_k,
        choose_best_k,
        clean_correlation_distance_vector,
        extract_visual_feature_frame,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "nsd_1000"
ROIS = ("ffa", "ppa", "eba")
VOXEL_COUNTS = {"ffa": 716, "ppa": 753, "eba": 2045}
REFERENCE_COUNT = 185
EXPECTED_MODEL_COUNT = 26
SCHEMA_VERSION = 1

PCA_DIM = 50
K_MIN = 2
K_MAX = 5
RANDOM_STATE = 42
KMEANS_N_INIT = 20
HIERARCHICAL_LINKAGE = "average"
HIERARCHICAL_DISTANCE = "correlation"

# Raw float64 means reproduce the checked-in aggregate to machine precision.
# The small explicit tolerance allows portable CSV parsing without accepting a
# scientifically meaningful discrepancy.
MEAN_ABSOLUTE_TOLERANCE = 1e-10
MEAN_RELATIVE_TOLERANCE = 1e-10

ID_COLUMN_NAMES = {
    "image_name",
    "filename",
    "file_name",
    "image",
    "image_id",
    "img",
    "img_id",
    "stimulus",
    "stimulus_id",
}

ARRAY_CONTRACT = {
    "image_ids": "Canonical ordered reference image identifiers (Unicode)",
    "reference_means": "Authoritative raw float64 row means",
    "reference_voxels": "Raw reference voxel matrix stored as float32",
    "imputer_statistics": "Per-voxel SimpleImputer mean statistics (float64)",
    "scaler_mean": "Per-voxel StandardScaler means (float64)",
    "scaler_scale": "Per-voxel StandardScaler scales (float64)",
    "cluster_pca_mean": "Cluster-space PCA feature means (float64)",
    "cluster_pca_components": "Cluster-space PCA components (float64)",
    "cluster_reference_coordinates": "Fixed reference coordinates in cluster PCA space (float64)",
    "kmeans_centroids": "Fixed K-means centroids in cluster PCA space (float64)",
    "kmeans_labels": "Fixed reference K-means labels (int32)",
    "plot_pca_mean": "Two-dimensional plotting PCA feature means (float64)",
    "plot_pca_components": "Two-dimensional plotting PCA components (float64)",
    "plot_reference_coordinates": "Fixed two-dimensional reference coordinates (float64)",
    "standardized_reference_matrix": "Imputed and z-scored reference voxel matrix (float32)",
    "hierarchical_labels": "Fixed reference hierarchical labels (int32)",
    "hierarchical_medoid_indices": "Reference row index for each hierarchical cluster medoid (int32)",
    "hierarchical_medoid_ids": "Reference image identifier for each hierarchical cluster medoid (Unicode)",
    "hierarchical_linkage": "Fixed average-linkage hierarchy (float64)",
}

VISUAL_FEATURE_SCHEMA = "infoviz_visual_77_v1"
VISUAL_ARRAY_CONTRACT = {
    "feature_names": "Exact ordered names of the 77 hand-built visual features (Unicode)",
    "image_ids": "Canonical ordered reference image identifiers (Unicode)",
    "reference_features": "Authoritative reference visual features (float64)",
    "scaler_mean": "Per-feature StandardScaler means (float64)",
    "scaler_scale": "Per-feature StandardScaler scales (float64)",
    "plot_pca_mean": "Two-dimensional visual PCA feature means (float64)",
    "plot_pca_components": "Two-dimensional visual PCA components (float64)",
    "plot_reference_coordinates": "Fixed two-dimensional reference coordinates (float64)",
    "kmeans_centroids": "Fixed visual K-means centroids sign-aligned to checked plotting PCA axes (float64)",
    "kmeans_labels": "Fixed visual reference K-means labels (int32)",
    "standardized_reference_matrix": "Z-scored visual reference feature matrix (float64)",
    "hierarchical_labels": "Fixed visual hierarchical labels (int32)",
    "hierarchical_medoid_indices": "Reference row index for each visual hierarchy medoid (int32)",
    "hierarchical_medoid_ids": "Reference image identifier for each visual hierarchy medoid (Unicode)",
    "hierarchical_linkage": "Fixed visual average-linkage hierarchy (float64)",
}


class ReferenceValidationError(ValueError):
    """Raised when an input cannot be part of a scientifically valid bundle."""


@dataclass(frozen=True)
class SourceCandidate:
    model: str
    roi: str
    timestamp: str
    path: Path


@dataclass(frozen=True)
class ValidatedSource:
    candidate: SourceCandidate
    image_ids: tuple[str, ...]
    voxel_columns: tuple[str, ...]
    voxels: np.ndarray
    means: np.ndarray
    sha256: str


@dataclass(frozen=True)
class StaticClusterContract:
    plot_coordinates: np.ndarray
    kmeans_labels: np.ndarray
    hierarchical_labels: np.ndarray
    kmeans_file: Path
    kmeans_sha256: str
    hierarchical_file: Path
    hierarchical_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_image_ids(count: int = REFERENCE_COUNT) -> tuple[str, ...]:
    return tuple(f"image_{index:03d}.png" for index in range(1, count + 1))


def discover_source_candidates(
    input_dir: Path,
    *,
    expected_model_count: int = EXPECTED_MODEL_COUNT,
    required_rois: Sequence[str] = ROIS,
) -> dict[tuple[str, str], list[SourceCandidate]]:
    """Discover strict filename matches and group timestamped duplicates.

    Non-CSV files are outside the source contract and are ignored.  Every CSV in
    the raw source directory must match the canonical NSD-1000 filename schema.
    """
    if not input_dir.is_dir():
        raise ReferenceValidationError(f"Raw source directory does not exist: {input_dir}")

    groups: dict[tuple[str, str], list[SourceCandidate]] = {}
    unmatched: list[str] = []
    for path in sorted(input_dir.glob("*.csv"), key=lambda item: item.name):
        match = FILENAME_RE.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        model = match.group("model")
        roi = match.group("roi").lower()
        timestamp = match.group("timestamp")
        candidate = SourceCandidate(model=model, roi=roi, timestamp=timestamp, path=path)
        groups.setdefault((model, roi), []).append(candidate)

    if unmatched:
        raise ReferenceValidationError(
            "CSV filenames must encode murtylab_<model>_nsd_1000_<roi>_<timestamp>Z.csv; "
            f"unmatched: {', '.join(unmatched)}"
        )
    if not groups:
        raise ReferenceValidationError(f"No canonical NSD-1000 CSV files found in {input_dir}")

    models = sorted({model for model, _ in groups})
    if len(models) != expected_model_count:
        raise ReferenceValidationError(
            f"Expected exactly {expected_model_count} supported models, found {len(models)}: "
            + ", ".join(models)
        )

    expected_roi_set = set(required_rois)
    triad_errors: list[str] = []
    for model in models:
        found_rois = {roi for candidate_model, roi in groups if candidate_model == model}
        if found_rois != expected_roi_set:
            missing = sorted(expected_roi_set - found_rois)
            unexpected = sorted(found_rois - expected_roi_set)
            triad_errors.append(
                f"{model}: missing={missing or 'none'}, unexpected={unexpected or 'none'}"
            )
    if triad_errors:
        raise ReferenceValidationError(
            "Every supported model must have the complete FFA/PPA/EBA triad; "
            + "; ".join(triad_errors)
        )

    for candidates in groups.values():
        candidates.sort(key=lambda item: (item.timestamp, item.path.name), reverse=True)
    return groups


def load_combined_means(
    path: Path,
    models: Sequence[str],
    *,
    expected_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    try:
        combined = pd.read_csv(path)
    except Exception as exc:  # pandas parser errors differ by release.
        raise ReferenceValidationError(f"Could not read combined means {path}: {exc}") from exc

    expected_columns = ["image_name"] + [
        f"{model}_{roi}" for model in sorted(models) for roi in ROIS
    ]
    if combined.columns.tolist() != expected_columns:
        missing = sorted(set(expected_columns) - set(combined.columns))
        unexpected = sorted(set(combined.columns) - set(expected_columns))
        raise ReferenceValidationError(
            "Combined means columns/order do not match the discovered model/ROI contract; "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )

    ids = combined["image_name"]
    if ids.isna().any():
        raise ReferenceValidationError("Combined means contains a missing image identifier")
    image_ids = tuple(ids.astype(str).tolist())
    if len(image_ids) != REFERENCE_COUNT:
        raise ReferenceValidationError(
            f"Combined means must contain {REFERENCE_COUNT} rows, found {len(image_ids)}"
        )
    if len(set(image_ids)) != len(image_ids):
        raise ReferenceValidationError("Combined means contains duplicate image identifiers")
    required_ids = tuple(expected_ids) if expected_ids is not None else canonical_image_ids()
    if image_ids != required_ids:
        raise ReferenceValidationError(
            "Combined means image identifiers/order do not match the canonical reference manifest"
        )

    for column in expected_columns[1:]:
        try:
            numeric = pd.to_numeric(combined[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ReferenceValidationError(
                f"Combined means column {column!r} is not entirely numeric"
            ) from exc
        if not np.isfinite(numeric).all():
            raise ReferenceValidationError(
                f"Combined means column {column!r} contains non-finite values"
            )
        combined[column] = numeric
    return combined


def _strict_id_column(columns: Iterable[Any], path: Path) -> str:
    column_names = [str(column) for column in columns]
    matches = [column for column in column_names if column.lower() in ID_COLUMN_NAMES]
    if len(matches) != 1:
        raise ReferenceValidationError(
            f"{path.name}: expected exactly one recognized image ID column, found {matches}"
        )
    return matches[0]


def load_and_validate_source(
    candidate: SourceCandidate,
    *,
    expected_ids: Sequence[str],
    expected_voxel_count: int,
    expected_means: np.ndarray,
    mean_atol: float = MEAN_ABSOLUTE_TOLERANCE,
    mean_rtol: float = MEAN_RELATIVE_TOLERANCE,
) -> ValidatedSource:
    path = candidate.path
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ReferenceValidationError(f"{path.name}: CSV parse failed: {exc}") from exc

    if not frame.columns.is_unique:
        raise ReferenceValidationError(f"{path.name}: duplicate CSV column names are not allowed")
    id_column = _strict_id_column(frame.columns, path)
    if frame[id_column].isna().any():
        raise ReferenceValidationError(f"{path.name}: missing image identifier")
    image_ids = tuple(frame[id_column].astype(str).tolist())
    if len(image_ids) != len(expected_ids):
        raise ReferenceValidationError(
            f"{path.name}: expected {len(expected_ids)} rows, found {len(image_ids)}"
        )
    if len(set(image_ids)) != len(image_ids):
        raise ReferenceValidationError(f"{path.name}: duplicate image identifiers are not allowed")
    if image_ids != tuple(expected_ids):
        raise ReferenceValidationError(
            f"{path.name}: image identifiers/order do not match the canonical reference manifest"
        )

    voxel_columns = tuple(str(column) for column in frame.columns if str(column) != id_column)
    unnamed = [column for column in voxel_columns if column.startswith("Unnamed")]
    if unnamed:
        raise ReferenceValidationError(
            f"{path.name}: unnamed/implicit columns are not allowed: {unnamed}"
        )
    if len(voxel_columns) != expected_voxel_count:
        raise ReferenceValidationError(
            f"{path.name}: expected {expected_voxel_count} voxel columns for "
            f"{candidate.roi.upper()}, found {len(voxel_columns)}"
        )

    numeric_columns: list[np.ndarray] = []
    for column in voxel_columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ReferenceValidationError(
                f"{path.name}: voxel column {column!r} contains a non-numeric value"
            ) from exc
        if not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values))[:5].tolist()
            raise ReferenceValidationError(
                f"{path.name}: voxel column {column!r} contains non-finite values "
                f"at row indices {bad_rows}"
            )
        numeric_columns.append(values)

    voxels = np.column_stack(numeric_columns).astype(np.float64, copy=False)
    means = voxels.mean(axis=1, dtype=np.float64)
    if not np.allclose(means, expected_means, atol=mean_atol, rtol=mean_rtol):
        differences = np.abs(means - expected_means)
        worst = int(np.argmax(differences))
        raise ReferenceValidationError(
            f"{path.name}: row means do not reproduce combined_image_means.csv "
            f"(max_abs_error={differences[worst]:.17g}, image={image_ids[worst]}, "
            f"atol={mean_atol:g}, rtol={mean_rtol:g})"
        )

    return ValidatedSource(
        candidate=candidate,
        image_ids=image_ids,
        voxel_columns=voxel_columns,
        voxels=voxels,
        means=means,
        sha256=sha256_file(path),
    )


def select_newest_valid_sources(
    candidates: Mapping[tuple[str, str], Sequence[SourceCandidate]],
    combined: pd.DataFrame,
    *,
    voxel_counts: Mapping[str, int] = VOXEL_COUNTS,
) -> tuple[dict[tuple[str, str], ValidatedSource], dict[str, Any]]:
    """Select the newest valid duplicate deterministically and record decisions."""
    expected_ids = tuple(combined["image_name"].astype(str).tolist())
    selected: dict[tuple[str, str], ValidatedSource] = {}
    decisions: dict[str, Any] = {}
    failures: list[str] = []

    for model, roi in sorted(candidates, key=lambda key: (key[0], ROIS.index(key[1]))):
        key = (model, roi)
        expected_means = combined[f"{model}_{roi}"].to_numpy(dtype=np.float64)
        rejected: list[dict[str, str]] = []
        valid_source: ValidatedSource | None = None
        for candidate in candidates[key]:
            try:
                valid_source = load_and_validate_source(
                    candidate,
                    expected_ids=expected_ids,
                    expected_voxel_count=voxel_counts[roi],
                    expected_means=expected_means,
                )
                break
            except ReferenceValidationError as exc:
                rejected.append({"file": candidate.path.name, "reason": str(exc)})

        decision_key = f"{model}/{roi}"
        if valid_source is None:
            failures.append(
                f"{decision_key}: no valid candidate ({'; '.join(item['reason'] for item in rejected)})"
            )
            continue
        selected[key] = valid_source
        decisions[decision_key] = {
            "selected": valid_source.candidate.path.name,
            "selected_timestamp": valid_source.candidate.timestamp,
            "candidate_count": len(candidates[key]),
            "rejected_newer_candidates": rejected,
        }

    if failures:
        raise ReferenceValidationError("Source validation failed:\n- " + "\n- ".join(failures))
    return selected, decisions


def _load_static_cluster_frame(
    path: Path,
    *,
    expected_ids: Sequence[str],
    required_columns: Sequence[str],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ReferenceValidationError(f"Could not read checked static cluster file {path}: {exc}") from exc
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ReferenceValidationError(f"{path.name}: missing static cluster columns {missing}")
    image_ids = tuple(frame["image_name"].astype(str).tolist())
    if image_ids != tuple(expected_ids):
        raise ReferenceValidationError(
            f"{path.name}: static cluster image identifiers/order do not match raw references"
        )
    return frame


def load_static_voxel_cluster_contracts(
    selected: Mapping[tuple[str, str], ValidatedSource],
    *,
    kmeans_dir: Path,
    hierarchical_dir: Path,
) -> dict[tuple[str, str], StaticClusterContract]:
    """Load checked plot axes/labels used by the existing standalone InfoViz UI."""
    contracts: dict[tuple[str, str], StaticClusterContract] = {}
    for key, source in selected.items():
        result_name = f"{source.candidate.path.stem}_clusters.csv"
        kmeans_file = kmeans_dir / result_name
        hierarchical_file = hierarchical_dir / result_name
        kmeans = _load_static_cluster_frame(
            kmeans_file,
            expected_ids=source.image_ids,
            required_columns=("image_name", "cluster_label", "pca_x", "pca_y"),
        )
        hierarchical = _load_static_cluster_frame(
            hierarchical_file,
            expected_ids=source.image_ids,
            required_columns=("image_name", "cluster_label"),
        )
        plot_coordinates = kmeans[["pca_x", "pca_y"]].to_numpy(dtype=np.float64)
        kmeans_labels = pd.to_numeric(kmeans["cluster_label"], errors="raise").to_numpy(
            dtype=np.int32
        )
        hierarchical_labels = pd.to_numeric(
            hierarchical["cluster_label"], errors="raise"
        ).to_numpy(dtype=np.int32)
        if not np.isfinite(plot_coordinates).all():
            raise ReferenceValidationError(f"{kmeans_file.name}: non-finite static PCA coordinate")
        contracts[key] = StaticClusterContract(
            plot_coordinates=plot_coordinates,
            kmeans_labels=kmeans_labels,
            hierarchical_labels=hierarchical_labels,
            kmeans_file=kmeans_file,
            kmeans_sha256=sha256_file(kmeans_file),
            hierarchical_file=hierarchical_file,
            hierarchical_sha256=sha256_file(hierarchical_file),
        )
    return contracts


def load_static_visual_cluster_contract(
    *,
    kmeans_file: Path,
    hierarchical_file: Path,
    expected_ids: Sequence[str],
) -> StaticClusterContract:
    kmeans = _load_static_cluster_frame(
        kmeans_file,
        expected_ids=expected_ids,
        required_columns=("image_name", "cluster_label", "plot_x", "plot_y"),
    )
    hierarchical = _load_static_cluster_frame(
        hierarchical_file,
        expected_ids=expected_ids,
        required_columns=("image_name", "cluster_label"),
    )
    plot_coordinates = kmeans[["plot_x", "plot_y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(plot_coordinates).all():
        raise ReferenceValidationError(f"{kmeans_file.name}: non-finite visual plot coordinate")
    return StaticClusterContract(
        plot_coordinates=plot_coordinates,
        kmeans_labels=pd.to_numeric(kmeans["cluster_label"], errors="raise").to_numpy(
            dtype=np.int32
        ),
        hierarchical_labels=pd.to_numeric(
            hierarchical["cluster_label"], errors="raise"
        ).to_numpy(dtype=np.int32),
        kmeans_file=kmeans_file,
        kmeans_sha256=sha256_file(kmeans_file),
        hierarchical_file=hierarchical_file,
        hierarchical_sha256=sha256_file(hierarchical_file),
    )


def hierarchical_medoid_indices(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    medoids: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) == 1:
            medoids.append(int(members[0]))
            continue
        within = distance_matrix[np.ix_(members, members)]
        local_index = int(np.argmin(within.sum(axis=1)))
        medoids.append(int(members[local_index]))
    return np.asarray(medoids, dtype=np.int32)


def align_plot_projection_to_static(
    arrays: Mapping[str, np.ndarray],
    static_coordinates: np.ndarray,
    *,
    context: str,
    residual_tolerance: float,
    align_kmeans_centroids: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Resolve PCA sign ambiguity against checked axes, without moving references."""
    computed = np.asarray(arrays["plot_reference_coordinates"], dtype=np.float64)
    target = np.asarray(static_coordinates, dtype=np.float64)
    if computed.shape != target.shape or computed.ndim != 2 or computed.shape[1] != 2:
        raise ReferenceValidationError(
            f"{context}: computed/static plot coordinate shapes differ: "
            f"{computed.shape} vs {target.shape}"
        )

    signs: list[int] = []
    residuals: list[float] = []
    for axis in range(2):
        positive_error = float(np.max(np.abs(computed[:, axis] - target[:, axis])))
        negative_error = float(np.max(np.abs(-computed[:, axis] - target[:, axis])))
        sign = 1 if positive_error <= negative_error else -1
        signs.append(sign)
        residuals.append(min(positive_error, negative_error))
    max_residual = max(residuals)
    if max_residual > residual_tolerance:
        raise ReferenceValidationError(
            f"{context}: saved PCA axes are not equivalent to checked static axes by sign "
            f"(max_abs_error={max_residual:.17g}, tolerance={residual_tolerance:g})"
        )

    aligned = dict(arrays)
    sign_array = np.asarray(signs, dtype=np.float64)
    aligned["plot_pca_components"] = (
        np.asarray(arrays["plot_pca_components"], dtype=np.float64)
        * sign_array[:, np.newaxis]
    )
    if align_kmeans_centroids:
        centroids = np.asarray(arrays["kmeans_centroids"], dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            raise ReferenceValidationError(
                f"{context}: visual K-means centroids must have shape (k, 2), "
                f"found {centroids.shape}"
            )
        aligned["kmeans_centroids"] = centroids * sign_array[np.newaxis, :]
    # These are the historical points already rendered by InfoViz.  Store them
    # exactly; only query points are transformed through the sign-aligned PCA.
    aligned["plot_reference_coordinates"] = target.copy()
    return aligned, {
        "plot_axis_signs": signs,
        "plot_alignment_max_abs_error": max_residual,
        "plot_alignment_policy": "checked_static_axes_sign_only",
        "kmeans_centroids_aligned_with_plot_axes": align_kmeans_centroids,
    }


def fit_reference_arrays(
    source: ValidatedSource,
    *,
    pca_dim: int = PCA_DIM,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
    linkage_method: str = HIERARCHICAL_LINKAGE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the exact current voxel clustering paths and retain transform state."""
    imputer = SimpleImputer(strategy="mean")
    imputed = imputer.fit_transform(source.voxels)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputed)

    used_pca_dim = max(2, min(pca_dim, standardized.shape[0] - 1, standardized.shape[1]))
    cluster_pca = PCA(n_components=used_pca_dim, random_state=random_state)
    cluster_coordinates = cluster_pca.fit_transform(standardized)
    kmeans = choose_best_k(
        cluster_coordinates,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    if kmeans is None:
        raise ReferenceValidationError(
            f"Could not find a valid K-means clustering for {source.candidate.path.name}"
        )

    plot_pca = PCA(n_components=2, random_state=random_state)
    plot_coordinates = plot_pca.fit_transform(standardized)

    correlation_vector = clean_correlation_distance_vector(
        pdist(standardized, metric=HIERARCHICAL_DISTANCE)
    )
    correlation_matrix = squareform(correlation_vector)
    hierarchy = linkage(correlation_vector, method=linkage_method)
    hierarchical = choose_best_hierarchical_k(
        correlation_vector,
        hierarchy,
        k_min=k_min,
        k_max=k_max,
    )
    if hierarchical is None:
        raise ReferenceValidationError(
            f"Could not find a valid hierarchical clustering for {source.candidate.path.name}"
        )
    medoid_indices = hierarchical_medoid_indices(correlation_matrix, hierarchical["labels"])
    image_ids = np.asarray(source.image_ids, dtype=str)

    arrays = {
        "image_ids": image_ids,
        "reference_means": np.asarray(source.means, dtype=np.float64),
        "reference_voxels": np.asarray(source.voxels, dtype=np.float32),
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float64),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "cluster_pca_mean": np.asarray(cluster_pca.mean_, dtype=np.float64),
        "cluster_pca_components": np.asarray(cluster_pca.components_, dtype=np.float64),
        "cluster_reference_coordinates": np.asarray(cluster_coordinates, dtype=np.float64),
        "kmeans_centroids": np.asarray(kmeans["model"].cluster_centers_, dtype=np.float64),
        "kmeans_labels": np.asarray(kmeans["labels"], dtype=np.int32),
        "plot_pca_mean": np.asarray(plot_pca.mean_, dtype=np.float64),
        "plot_pca_components": np.asarray(plot_pca.components_, dtype=np.float64),
        "plot_reference_coordinates": np.asarray(plot_coordinates, dtype=np.float64),
        "standardized_reference_matrix": np.asarray(standardized, dtype=np.float32),
        "hierarchical_labels": np.asarray(hierarchical["labels"], dtype=np.int32),
        "hierarchical_medoid_indices": medoid_indices,
        "hierarchical_medoid_ids": image_ids[medoid_indices],
        "hierarchical_linkage": np.asarray(hierarchy, dtype=np.float64),
    }
    metadata = {
        "cluster_pca_dimensions": used_pca_dim,
        "kmeans_best_k": int(kmeans["k"]),
        "kmeans_silhouette": float(kmeans["silhouette"]),
        "hierarchical_best_k": int(hierarchical["k"]),
        "hierarchical_silhouette": float(hierarchical["silhouette"]),
    }
    return arrays, metadata


def load_visual_features(
    path: Path,
    *,
    expected_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Load the checked-in output of the canonical 77-feature extractor strictly."""
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ReferenceValidationError(f"Could not read visual features {path}: {exc}") from exc
    if not frame.columns.is_unique:
        raise ReferenceValidationError(f"{path.name}: duplicate visual feature columns")
    if not len(frame.columns) or frame.columns[0] != "image_name":
        raise ReferenceValidationError(
            f"{path.name}: first column must be the canonical image_name identifier"
        )
    image_ids = tuple(frame["image_name"].astype(str).tolist())
    if image_ids != tuple(expected_ids):
        raise ReferenceValidationError(
            f"{path.name}: visual image identifiers/order do not match the reference manifest"
        )
    feature_names = tuple(str(column) for column in frame.columns[1:])
    if len(feature_names) != 77:
        raise ReferenceValidationError(
            f"{path.name}: {VISUAL_FEATURE_SCHEMA} requires 77 ordered features, "
            f"found {len(feature_names)}"
        )
    numeric_columns: list[np.ndarray] = []
    for column in feature_names:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ReferenceValidationError(
                f"{path.name}: visual feature {column!r} contains a non-numeric value"
            ) from exc
        if not np.isfinite(values).all():
            raise ReferenceValidationError(
                f"{path.name}: visual feature {column!r} contains non-finite values"
            )
        numeric_columns.append(values)
    return feature_names, np.column_stack(numeric_columns).astype(np.float64, copy=False)


def extract_and_validate_visual_features(
    *,
    image_dir: Path,
    checked_features_path: Path,
    expected_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    """Run the existing extractor and verify its output against checked-in features."""
    checked_names, checked_values = load_visual_features(
        checked_features_path,
        expected_ids=expected_ids,
    )
    try:
        extracted = extract_visual_feature_frame(image_dir, "all")
    except Exception as exc:
        raise ReferenceValidationError(
            f"Could not extract canonical visual features from {image_dir}: {exc}"
        ) from exc

    if extracted.columns[0] != "image_name":
        raise ReferenceValidationError("Visual extractor did not return image_name first")
    extracted_ids = tuple(extracted["image_name"].astype(str).tolist())
    extracted_names = tuple(str(column) for column in extracted.columns[1:])
    if extracted_ids != tuple(expected_ids):
        raise ReferenceValidationError(
            "Visual extractor image identifiers/order do not match the reference manifest"
        )
    if extracted_names != checked_names:
        raise ReferenceValidationError(
            "Visual extractor feature names/order do not match the checked-in 77-feature schema"
        )
    extracted_values = extracted[list(extracted_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(extracted_values).all():
        raise ReferenceValidationError("Visual extractor returned non-finite values")
    # The table predates a small float32 angular arithmetic change in the current
    # extractor.  It remains a strict schema/reference check while allowing only
    # the measured angular drift (<=2 histogram degrees, <=5e-5 hue degrees).
    differences = np.abs(extracted_values - checked_values)
    tolerances = np.full(len(extracted_names), 1e-7, dtype=np.float64)
    for index, name in enumerate(extracted_names):
        if "hue" in name:
            tolerances[index] = 5e-5
        elif name == "dominant_angle":
            tolerances[index] = 2.0
    invalid = differences > tolerances[np.newaxis, :]
    if invalid.any():
        masked_differences = np.where(invalid, differences, -1.0)
        row, column = np.unravel_index(
            int(np.argmax(masked_differences)),
            masked_differences.shape,
        )
        raise ReferenceValidationError(
            "Extracted visual features exceed the documented checked-table drift "
            f"(abs_error={differences[row, column]:.17g}, tolerance={tolerances[column]:g}, "
            f"image={extracted_ids[row]}, feature={extracted_names[column]})"
        )

    image_hashes = {
        image_id: sha256_file(image_dir / image_id)
        for image_id in expected_ids
    }
    return extracted_names, extracted_values, image_hashes


def fit_visual_reference_arrays(
    *,
    image_ids: Sequence[str],
    feature_names: Sequence[str],
    reference_features: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
    linkage_method: str = HIERARCHICAL_LINKAGE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce visual PCA/K-means and correlation hierarchy without t-SNE."""
    scaler = StandardScaler()
    standardized = scaler.fit_transform(reference_features)
    plot_pca = PCA(n_components=2, random_state=random_state)
    plot_coordinates = plot_pca.fit_transform(standardized)

    kmeans = choose_best_k(
        plot_coordinates,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    if kmeans is None:
        raise ReferenceValidationError("Could not find a valid visual K-means clustering")

    correlation_vector = clean_correlation_distance_vector(
        pdist(standardized, metric=HIERARCHICAL_DISTANCE)
    )
    correlation_matrix = squareform(correlation_vector)
    hierarchy = linkage(correlation_vector, method=linkage_method)
    hierarchical = choose_best_hierarchical_k(
        correlation_vector,
        hierarchy,
        k_min=k_min,
        k_max=k_max,
    )
    if hierarchical is None:
        raise ReferenceValidationError("Could not find a valid hierarchical visual clustering")
    medoid_indices = hierarchical_medoid_indices(correlation_matrix, hierarchical["labels"])
    image_id_array = np.asarray(image_ids, dtype=str)

    arrays = {
        "feature_names": np.asarray(feature_names, dtype=str),
        "image_ids": image_id_array,
        "reference_features": np.asarray(reference_features, dtype=np.float64),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "plot_pca_mean": np.asarray(plot_pca.mean_, dtype=np.float64),
        "plot_pca_components": np.asarray(plot_pca.components_, dtype=np.float64),
        "plot_reference_coordinates": np.asarray(plot_coordinates, dtype=np.float64),
        "kmeans_centroids": np.asarray(kmeans["model"].cluster_centers_, dtype=np.float64),
        "kmeans_labels": np.asarray(kmeans["labels"], dtype=np.int32),
        "standardized_reference_matrix": np.asarray(standardized, dtype=np.float64),
        "hierarchical_labels": np.asarray(hierarchical["labels"], dtype=np.int32),
        "hierarchical_medoid_indices": medoid_indices,
        "hierarchical_medoid_ids": image_id_array[medoid_indices],
        "hierarchical_linkage": np.asarray(hierarchy, dtype=np.float64),
    }
    metadata = {
        "feature_schema": VISUAL_FEATURE_SCHEMA,
        "feature_count": len(feature_names),
        "kmeans_best_k": int(kmeans["k"]),
        "kmeans_silhouette": float(kmeans["silhouette"]),
        "hierarchical_best_k": int(hierarchical["k"]),
        "hierarchical_silhouette": float(hierarchical["silhouette"]),
    }
    return arrays, metadata


def write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    array_contract: Mapping[str, str] = ARRAY_CONTRACT,
) -> None:
    """Write compressed NumPy arrays with stable order and ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            if name not in array_contract:
                raise ValueError(f"Undocumented artifact array: {name}")
            array = np.asanyarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"Object dtype is not portable with allow_pickle=False: {name}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _library_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": pillow_version,
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _clustering_manifest() -> dict[str, Any]:
    return {
        "preprocessing": ["SimpleImputer(strategy=mean)", "StandardScaler"],
        "cluster_pca_dimensions": PCA_DIM,
        "kmeans": {
            "k_min": K_MIN,
            "k_max": K_MAX,
            "n_init": KMEANS_N_INIT,
            "random_state": RANDOM_STATE,
            "selection_metric": "euclidean_silhouette",
            "tie_policy": "first_lower_k",
        },
        "plot_projection": {
            "method": "PCA",
            "dimensions": 2,
            "random_state": RANDOM_STATE,
            "axis_alignment": "checked_static_axes_sign_only",
            "historical_coordinates": "preserved_exactly",
            "visual_kmeans_centroids": "aligned_with_plot_axes",
        },
        "hierarchical": {
            "distance": HIERARCHICAL_DISTANCE,
            "linkage": HIERARCHICAL_LINKAGE,
            "k_min": K_MIN,
            "k_max": K_MAX,
            "selection_metric": "precomputed_distance_silhouette",
            "tie_policy": "first_lower_k",
            "out_of_sample_policy": "nearest_medoid_assignment",
        },
    }


def _artifact_schema_manifest() -> dict[str, Any]:
    return {
        "format": "npz",
        "compression": "deflate_level_9",
        "allow_pickle": False,
        "arrays": ARRAY_CONTRACT,
    }


def _optional_artifact_schema_manifest() -> dict[str, Any]:
    return {
        "visual": {
            "format": "npz",
            "compression": "deflate_level_9",
            "allow_pickle": False,
            "feature_schema": VISUAL_FEATURE_SCHEMA,
            "arrays": VISUAL_ARRAY_CONTRACT,
        },
        "dreamsim": {
            "available": False,
            "reason": (
                "Disabled: the current runtime encoder returns 768 dimensions while the "
                "checked-in reference bundle has 1792 dimensions; checkpoint provenance "
                "must be verified before out-of-sample projection"
            ),
        },
    }


def _reference_identity(
    *,
    models: Sequence[str],
    image_ids: Sequence[str],
    combined_means_sha256: str,
    selected: Mapping[tuple[str, str], ValidatedSource],
    visual_features_sha256: str,
    visual_images_sha256: str,
    static_clusters_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "rois": list(ROIS),
        "supported_models": list(sorted(models)),
        "image_count": len(image_ids),
        "reference_count": len(image_ids),
        "image_ids_sha256": sha256_json(list(image_ids)),
        "voxel_counts": VOXEL_COUNTS,
        "combined_means_sha256": combined_means_sha256,
        "source_sha256": {
            model: {roi: selected[(model, roi)].sha256 for roi in ROIS}
            for model in sorted(models)
        },
        "clustering": _clustering_manifest(),
        "mean_validation": {
            "absolute_tolerance": MEAN_ABSOLUTE_TOLERANCE,
            "relative_tolerance": MEAN_RELATIVE_TOLERANCE,
            "calculation_dtype": "float64",
        },
        "artifact_schema": _artifact_schema_manifest(),
        "optional_artifact_schemas": _optional_artifact_schema_manifest(),
        "visual_features_sha256": visual_features_sha256,
        "visual_images_sha256": visual_images_sha256,
        "static_clusters_sha256": static_clusters_sha256,
        "library_versions": _library_versions(),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _install_staged_directory(staging: Path, output_dir: Path) -> None:
    backup = output_dir.with_name(f".{output_dir.name}-backup-{uuid.uuid4().hex}")
    moved_old = False
    try:
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_old = True
        os.replace(staging, output_dir)
    except Exception:
        if moved_old and not output_dir.exists() and backup.exists():
            os.replace(backup, output_dir)
        raise
    else:
        if moved_old:
            shutil.rmtree(backup)


def public_manifest_from_full(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the browser-safe subset; raw filenames, hashes, and paths stay private."""
    return {
        "schema_version": manifest["schema_version"],
        "reference_version": manifest["reference_version"],
        "checksum": manifest["checksum"],
        "dataset": manifest["dataset"],
        "rois": manifest["rois"],
        "supported_models": manifest["supported_models"],
        "image_count": manifest["image_count"],
        "reference_count": manifest["reference_count"],
        "voxel_counts": manifest["voxel_counts"],
        "clustering": {
            "plot_projection": manifest["clustering"]["plot_projection"],
            "hierarchical_out_of_sample_policy": manifest["clustering"]["hierarchical"][
                "out_of_sample_policy"
            ],
        },
        "generated_at": manifest["generated_at"],
        "optional_artifacts": {
            "visual": {
                "available": True,
                "feature_schema": manifest["optional_artifacts"]["visual"]["feature_schema"],
            },
            "dreamsim": {"available": False},
        },
    }


def build_reference_bundle(
    *,
    input_dir: Path,
    combined_means_path: Path,
    output_dir: Path,
    public_manifest_path: Path | None,
    visual_image_dir: Path = REPO_ROOT / "public" / "images",
    voxel_clusters_dir: Path = REPO_ROOT / "public" / "data" / "voxel_clusters",
    voxel_hierarchical_clusters_dir: Path = REPO_ROOT
    / "public"
    / "data"
    / "voxel_clusters_hierarchical_corr",
    visual_clusters_file: Path = REPO_ROOT
    / "public"
    / "data"
    / "visual_clusters"
    / "visual_clusters.csv",
    visual_hierarchical_clusters_file: Path = REPO_ROOT
    / "public"
    / "data"
    / "visual_clusters_hierarchical_corr"
    / "visual_clusters.csv",
    visual_features_path: Path | None = REPO_ROOT
    / "public"
    / "data"
    / "visual_clusters"
    / "visual_features_only.csv",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Validate all inputs, build all artifacts in staging, and install atomically."""
    candidates = discover_source_candidates(input_dir)
    models = sorted({model for model, _ in candidates})
    combined = load_combined_means(combined_means_path, models)
    selected, selection_decisions = select_newest_valid_sources(candidates, combined)
    static_voxel_contracts = load_static_voxel_cluster_contracts(
        selected,
        kmeans_dir=voxel_clusters_dir,
        hierarchical_dir=voxel_hierarchical_clusters_dir,
    )

    image_ids = tuple(combined["image_name"].astype(str).tolist())
    if visual_features_path is None:
        raise ReferenceValidationError(
            "The v1 comparison bundle requires the canonical visual feature artifact"
        )
    visual_feature_names, visual_features, visual_image_hashes = extract_and_validate_visual_features(
        image_dir=visual_image_dir,
        checked_features_path=visual_features_path,
        expected_ids=image_ids,
    )
    static_visual_contract = load_static_visual_cluster_contract(
        kmeans_file=visual_clusters_file,
        hierarchical_file=visual_hierarchical_clusters_file,
        expected_ids=image_ids,
    )
    combined_hash = sha256_file(combined_means_path)
    visual_features_hash = sha256_json(
        {
            "feature_names": list(visual_feature_names),
            "values_sha256": hashlib.sha256(
                np.ascontiguousarray(visual_features, dtype=np.float64).tobytes()
            ).hexdigest(),
            "checked_table_sha256": sha256_file(visual_features_path),
        }
    )
    visual_images_hash = sha256_json(visual_image_hashes)
    static_clusters_hash = sha256_json(
        {
            "voxel": {
                f"{model}/{roi}": {
                    "kmeans": contract.kmeans_sha256,
                    "hierarchical": contract.hierarchical_sha256,
                }
                for (model, roi), contract in sorted(static_voxel_contracts.items())
            },
            "visual": {
                "kmeans": static_visual_contract.kmeans_sha256,
                "hierarchical": static_visual_contract.hierarchical_sha256,
            },
        }
    )
    identity = _reference_identity(
        models=models,
        image_ids=image_ids,
        combined_means_sha256=combined_hash,
        selected=selected,
        visual_features_sha256=visual_features_hash,
        visual_images_sha256=visual_images_hash,
        static_clusters_sha256=static_clusters_hash,
    )
    reference_version = sha256_json(identity)
    generation_time = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-build-", dir=output_dir.parent))
    artifacts_manifest: dict[str, dict[str, Any]] = {}
    try:
        for model in models:
            artifacts_manifest[model] = {}
            for roi in ROIS:
                source = selected[(model, roi)]
                arrays, fitted_metadata = fit_reference_arrays(source)
                static_contract = static_voxel_contracts[(model, roi)]
                if not np.array_equal(arrays["kmeans_labels"], static_contract.kmeans_labels):
                    raise ReferenceValidationError(
                        f"{model}/{roi}: fitted K-means labels differ from checked static labels"
                    )
                if not np.array_equal(
                    arrays["hierarchical_labels"], static_contract.hierarchical_labels
                ):
                    raise ReferenceValidationError(
                        f"{model}/{roi}: fitted hierarchical labels differ from checked static labels"
                    )
                arrays, alignment_metadata = align_plot_projection_to_static(
                    arrays,
                    static_contract.plot_coordinates,
                    context=f"{model}/{roi}",
                    residual_tolerance=1e-9,
                )
                fitted_metadata.update(alignment_metadata)
                relative_path = Path("models") / model / f"{roi}.npz"
                artifact_path = staging / relative_path
                write_deterministic_npz(artifact_path, arrays)
                artifacts_manifest[model][roi] = {
                    "path": relative_path.as_posix(),
                    "sha256": sha256_file(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                    "source_file": source.candidate.path.name,
                    "source_timestamp": source.candidate.timestamp,
                    "source_sha256": source.sha256,
                    "static_kmeans_sha256": static_contract.kmeans_sha256,
                    "static_hierarchical_sha256": static_contract.hierarchical_sha256,
                    "image_count": len(source.image_ids),
                    "voxel_count": source.voxels.shape[1],
                    "arrays": {
                        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                        for name, array in sorted(arrays.items())
                    },
                    **fitted_metadata,
                }
                print(f"Built {model}/{roi}: {relative_path.as_posix()}", flush=True)

        visual_arrays, visual_metadata = fit_visual_reference_arrays(
            image_ids=image_ids,
            feature_names=visual_feature_names,
            reference_features=visual_features,
        )
        if not np.array_equal(
            visual_arrays["kmeans_labels"], static_visual_contract.kmeans_labels
        ):
            raise ReferenceValidationError(
                "Fitted visual K-means labels differ from checked static labels"
            )
        if not np.array_equal(
            visual_arrays["hierarchical_labels"], static_visual_contract.hierarchical_labels
        ):
            raise ReferenceValidationError(
                "Fitted visual hierarchical labels differ from checked static labels"
            )
        visual_arrays, visual_alignment_metadata = align_plot_projection_to_static(
            visual_arrays,
            static_visual_contract.plot_coordinates,
            context="visual",
            residual_tolerance=2e-3,
            align_kmeans_centroids=True,
        )
        visual_metadata.update(visual_alignment_metadata)
        visual_relative_path = Path("visual.npz")
        visual_artifact_path = staging / visual_relative_path
        write_deterministic_npz(
            visual_artifact_path,
            visual_arrays,
            array_contract=VISUAL_ARRAY_CONTRACT,
        )
        optional_artifacts = {
            "visual": {
                "available": True,
                "path": visual_relative_path.as_posix(),
                "sha256": sha256_file(visual_artifact_path),
                "size_bytes": visual_artifact_path.stat().st_size,
                "source_file": visual_features_path.name,
                "source_sha256": visual_features_hash,
                "source_images_sha256": visual_images_hash,
                "static_kmeans_sha256": static_visual_contract.kmeans_sha256,
                "static_hierarchical_sha256": static_visual_contract.hierarchical_sha256,
                "image_count": len(image_ids),
                "feature_schema": VISUAL_FEATURE_SCHEMA,
                "arrays": {
                    name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                    for name, array in sorted(visual_arrays.items())
                },
                **visual_metadata,
            },
            "dreamsim": identity["optional_artifact_schemas"]["dreamsim"],
        }
        print(f"Built visual reference artifact: {visual_relative_path.as_posix()}", flush=True)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "reference_version": reference_version,
            "checksum": reference_version,
            "dataset": DATASET,
            "rois": list(ROIS),
            "supported_models": models,
            "image_count": len(image_ids),
            "reference_count": len(image_ids),
            "image_ids_sha256": identity["image_ids_sha256"],
            "voxel_counts": VOXEL_COUNTS,
            "clustering": identity["clustering"],
            "mean_validation": identity["mean_validation"],
            "artifact_schema": identity["artifact_schema"],
            "optional_artifact_schemas": identity["optional_artifact_schemas"],
            "library_versions": identity["library_versions"],
            "source_selection": selection_decisions,
            "combined_means": {
                "file": combined_means_path.name,
                "sha256": combined_hash,
            },
            "artifacts": artifacts_manifest,
            "optional_artifacts": optional_artifacts,
            "generator_git": _git_state(REPO_ROOT),
            "generated_at": generation_time,
        }
        (staging / "manifest.json").write_text(_json_text(manifest), encoding="utf-8")
        _install_staged_directory(staging, output_dir)
        if public_manifest_path is not None:
            _atomic_write_text(
                public_manifest_path.resolve(),
                _json_text(public_manifest_from_full(manifest)),
            )
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual-image-dir",
        type=Path,
        default=REPO_ROOT / "public" / "images",
        help="Canonical 185 reference images used by the existing 77-feature extractor",
    )
    parser.add_argument(
        "--visual-features",
        type=Path,
        default=REPO_ROOT
        / "public"
        / "data"
        / "visual_clusters"
        / "visual_features_only.csv",
        help="Checked-in canonical 77-feature reference table",
    )
    parser.add_argument(
        "--voxel-clusters-dir",
        type=Path,
        default=REPO_ROOT / "public" / "data" / "voxel_clusters",
        help="Checked static voxel K-means outputs used to orient PCA plot axes",
    )
    parser.add_argument(
        "--voxel-hierarchical-clusters-dir",
        type=Path,
        default=REPO_ROOT / "public" / "data" / "voxel_clusters_hierarchical_corr",
        help="Checked static voxel hierarchical outputs used to validate labels",
    )
    parser.add_argument(
        "--visual-clusters-file",
        type=Path,
        default=REPO_ROOT
        / "public"
        / "data"
        / "visual_clusters"
        / "visual_clusters.csv",
        help="Checked static visual K-means output used to orient PCA plot axes",
    )
    parser.add_argument(
        "--visual-hierarchical-clusters-file",
        type=Path,
        default=REPO_ROOT
        / "public"
        / "data"
        / "visual_clusters_hierarchical_corr"
        / "visual_clusters.csv",
        help="Checked static visual hierarchical output used to validate labels",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "image_by_model",
        help="Directory containing raw model/ROI voxel CSV files",
    )
    parser.add_argument(
        "--combined-means",
        type=Path,
        default=REPO_ROOT / "public" / "data" / "combined_image_means.csv",
        help="Checked-in aggregate means used for strict validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reference",
        help="Generated server-side bundle directory (Git-ignored)",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=REPO_ROOT / "public" / "data" / "reference_manifest.json",
        help="Browser-safe manifest destination",
    )
    parser.add_argument(
        "--no-public-manifest",
        action="store_true",
        help="Do not update the browser-safe public manifest",
    )
    parser.add_argument(
        "--generated-at",
        help="Override ISO-8601 generation timestamp (useful for reproducibility tests)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = build_reference_bundle(
            input_dir=args.input_dir,
            combined_means_path=args.combined_means,
            output_dir=args.output_dir,
            public_manifest_path=None if args.no_public_manifest else args.public_manifest,
            visual_image_dir=args.visual_image_dir,
            voxel_clusters_dir=args.voxel_clusters_dir,
            voxel_hierarchical_clusters_dir=args.voxel_hierarchical_clusters_dir,
            visual_clusters_file=args.visual_clusters_file,
            visual_hierarchical_clusters_file=args.visual_hierarchical_clusters_file,
            visual_features_path=args.visual_features,
            generated_at=args.generated_at,
        )
    except ReferenceValidationError as exc:
        raise SystemExit(f"Reference artifact build failed:\n{exc}") from exc
    print(
        f"Built {len(manifest['supported_models']) * len(manifest['rois'])} artifacts at "
        f"{args.output_dir}\nreference_version={manifest['reference_version']}"
    )


if __name__ == "__main__":
    main()
