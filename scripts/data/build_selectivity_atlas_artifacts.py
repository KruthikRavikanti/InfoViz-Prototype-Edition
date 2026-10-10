#!/usr/bin/env python3
"""Build browser artifacts for the mentor's selectivity-based voxel atlas.

The raw response CSVs contain the 185 Murty stimuli as rows and predicted
responses for several NSD subjects as columns.  This builder uses only the
audited Subject 1 prefix from each ROI and clusters the FFA, EBA, and PPA
voxels jointly.  It deliberately keeps UMAP out of the clustering path: UMAP
coordinates are emitted only so the browser can render the fitted atlas.
Each model summary also reports six deterministic split-half adjusted Rand
indices as a limited stability diagnostic; this does not repeat the mentor's
original k-selection study.

The source files are selected from ``artifacts/reference/manifest.json``.  That
manifest is the repository's audited record of which timestamped CSV belongs
to every model/ROI pair and which SHA-256 digest it must have.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

try:  # Supports both package-style test imports and direct CLI execution.
    from .validate_model_registries import RAW_FILENAME_RE
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    from validate_model_registries import RAW_FILENAME_RE  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data/image_by_model"
DEFAULT_REFERENCE_MANIFEST = REPO_ROOT / "artifacts/reference/manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "public/data/voxel_selectivity_atlas"
BASELINE_ATLAS_MANIFEST_SHA256 = "7244cead1cc3a9d6fb0074590eed025889994d23eecce432271cb8b38b28881e"

DATASET = "nsd_1000"
RESPONSE_SPACE = "nsd_1000_subject1_floc_t7_v1"
STIMULUS_SET = "murty185"
RESPONSE_KIND = "model_predictions"
IMAGE_COUNT = 185
ROI_ORDER = ("ffa", "eba", "ppa")
ROI_SPECS: Mapping[str, Mapping[str, Any]] = {
    "ffa": {"label": "FFA", "responseCount": 156},
    "eba": {"label": "EBA", "responseCount": 518},
    "ppa": {"label": "PPA", "responseCount": 170},
}
VOXEL_COUNT = sum(int(ROI_SPECS[roi]["responseCount"]) for roi in ROI_ORDER)
SCHEMA_VERSION = 1
TOP_IMAGE_COUNT = 5
SPLIT_HALF_ITERATIONS = 6
SPLIT_HALF_RANDOM_SEED = 42
SPLIT_HALF_ARI_THRESHOLD = 0.7

K14_STABILITY_CAVEAT = (
    "The mentor reported k=14 as split-half stable (ARI >= 0.70) in the original "
    "analysis. The six deterministic split-half ARIs reported here are a limited "
    "diagnostic on these 185 predicted-response profiles, not a revalidation of the "
    "original stability-based k-selection analysis. Treat k=14 as a fixed exploratory "
    "resolution, not a newly validated optimum."
)


class SelectivityAtlasValidationError(ValueError):
    """Raised when an input violates the audited atlas contract."""


@dataclass(frozen=True)
class AtlasConfig:
    pca_components: int = 35
    cluster_count: int = 14
    kmeans_n_init: int = 25
    kmeans_random_state: int = 0
    umap_neighbors: int = 25
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 42


@dataclass(frozen=True)
class SubjectRoiSource:
    model: str
    roi: str
    path: Path
    sha256: str
    image_ids: tuple[str, ...]
    responses: np.ndarray


@dataclass(frozen=True)
class AtlasFit:
    z_profiles: np.ndarray
    selectivity_features: np.ndarray
    pca_explained_variance_ratio_sum: float
    labels: np.ndarray
    embedding: np.ndarray
    distances_to_centroid: np.ndarray
    silhouette_by_voxel: np.ndarray
    silhouette: float
    split_half_stability: Mapping[str, Any]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_image_ids(count: int = IMAGE_COUNT) -> tuple[str, ...]:
    return tuple(f"image_{index:03d}.png" for index in range(1, count + 1))


def method_metadata(config: AtlasConfig) -> dict[str, Any]:
    """Return the browser-facing, explicit analysis contract."""
    return {
        "id": "mentor-selectivity-k14-v1",
        "objectClustered": "voxels",
        "normalization": "per-voxel z-score across images (ddof=0, epsilon=1e-9)",
        "pcaComponents": config.pca_components,
        "l2Normalization": True,
        "clustering": "K-means in L2-normalized PCA selectivity space",
        "clusterCount": config.cluster_count,
        "kmeansNInit": config.kmeans_n_init,
        "kmeansRandomState": config.kmeans_random_state,
        "embedding": (
            f"3D UMAP (n_neighbors={config.umap_neighbors}, "
            f"min_dist={config.umap_min_dist:g}, metric={config.umap_metric}, "
            f"random_state={config.umap_random_state}); display only"
        ),
        "embeddingUsedForClustering": False,
    }


def provenance_metadata() -> dict[str, Any]:
    """Name the stimulus and response spaces and rule out anatomical UMAP claims."""
    return {
        "stimulusSet": STIMULUS_SET,
        "responseSpace": RESPONSE_SPACE,
        "responseKind": RESPONSE_KIND,
        "embeddingIsAnatomical": False,
    }


def validate_subject1_columns(
    voxel_columns: Sequence[str], expected_count: int
) -> tuple[int, ...]:
    """Validate the exact contiguous ``s1_0..s1_N`` prefix and return indices."""
    columns = tuple(str(column) for column in voxel_columns)
    expected = tuple(f"s1_{index}" for index in range(expected_count))
    subject_columns = tuple(column for column in columns if column.startswith("s1_"))
    if columns[:expected_count] != expected or subject_columns != expected:
        first_mismatch = next(
            (
                index
                for index, (actual, wanted) in enumerate(
                    zip(columns[:expected_count], expected)
                )
                if actual != wanted
            ),
            min(len(columns), expected_count),
        )
        actual = columns[first_mismatch] if first_mismatch < len(columns) else "<missing>"
        wanted = expected[first_mismatch] if first_mismatch < len(expected) else "<none>"
        raise SelectivityAtlasValidationError(
            "Subject 1 voxel columns must be the exact contiguous leading prefix "
            f"s1_0..s1_{expected_count - 1}; first mismatch at {first_mismatch}: "
            f"expected {wanted!r}, found {actual!r}"
        )
    return tuple(range(expected_count))


def _validate_source_filename(path: Path, model: str, roi: str) -> None:
    match = RAW_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise SelectivityAtlasValidationError(
            f"Source filename does not match the audited NSD-1000 schema: {path.name}"
        )
    if match.group("model") != model or match.group("roi") != roi:
        raise SelectivityAtlasValidationError(
            f"Source {path.name} encodes {match.group('model')}/{match.group('roi')}, "
            f"not requested {model}/{roi}"
        )


def load_subject1_csv(
    path: Path,
    *,
    model: str,
    roi: str,
    expected_count: int,
    expected_image_ids: Sequence[str],
    expected_sha256: str | None = None,
) -> SubjectRoiSource:
    """Load and validate one audited ROI CSV without retaining other subjects."""
    if not path.is_file():
        raise SelectivityAtlasValidationError(f"Missing source CSV: {path}")
    _validate_source_filename(path, model, roi)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SelectivityAtlasValidationError(
            f"{path.name} SHA-256 {digest} does not match audited digest {expected_sha256}"
        )

    image_ids: list[str] = []
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise SelectivityAtlasValidationError(f"Source CSV is empty: {path}") from error
        if not header or header[0] != "image":
            raise SelectivityAtlasValidationError(
                f"{path.name}: first column must be exactly 'image'"
            )
        if len(header) != len(set(header)):
            raise SelectivityAtlasValidationError(
                f"{path.name}: duplicate column names are not allowed"
            )
        voxel_columns = tuple(header[1:])
        selected = validate_subject1_columns(voxel_columns, expected_count)

        for row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise SelectivityAtlasValidationError(
                    f"{path.name}: row {row_index + 2} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            image_ids.append(row[0])
            try:
                values = [float(row[column_index + 1]) for column_index in selected]
            except ValueError as error:
                raise SelectivityAtlasValidationError(
                    f"{path.name}: non-numeric Subject 1 response at row {row_index + 2}"
                ) from error
            if not np.isfinite(values).all():
                raise SelectivityAtlasValidationError(
                    f"{path.name}: non-finite Subject 1 response at row {row_index + 2}"
                )
            rows.append(values)

    expected_ids = tuple(str(value) for value in expected_image_ids)
    if tuple(image_ids) != expected_ids:
        raise SelectivityAtlasValidationError(
            f"{path.name}: image identifiers/order do not match the reference contract"
        )
    responses = np.asarray(rows, dtype=np.float64)
    expected_shape = (len(expected_ids), expected_count)
    if responses.shape != expected_shape:
        raise SelectivityAtlasValidationError(
            f"{path.name}: Subject 1 response shape {responses.shape}; expected {expected_shape}"
        )
    return SubjectRoiSource(
        model=model,
        roi=roi,
        path=path,
        sha256=digest,
        image_ids=tuple(image_ids),
        responses=responses,
    )


def load_reference_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectivityAtlasValidationError(f"Missing audited reference manifest: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectivityAtlasValidationError(
            f"Could not read audited reference manifest {path}: {error}"
        ) from error
    if document.get("dataset") != DATASET:
        raise SelectivityAtlasValidationError(
            f"Reference manifest dataset must be {DATASET!r}"
        )
    if document.get("image_count") != IMAGE_COUNT:
        raise SelectivityAtlasValidationError(
            f"Reference manifest must contain {IMAGE_COUNT} images"
        )
    models = document.get("supported_models")
    if not isinstance(models, list) or not models or len(models) != len(set(models)):
        raise SelectivityAtlasValidationError(
            "Reference manifest supported_models must be a non-empty unique list"
        )
    if not isinstance(document.get("source_selection"), dict) or not isinstance(
        document.get("artifacts"), dict
    ):
        raise SelectivityAtlasValidationError(
            "Reference manifest is missing audited source_selection/artifacts records"
        )
    return document


def load_model_sources(
    *,
    model: str,
    raw_dir: Path,
    reference_manifest: Mapping[str, Any],
    expected_image_ids: Sequence[str] | None = None,
) -> dict[str, SubjectRoiSource]:
    image_ids = tuple(expected_image_ids or canonical_image_ids())
    sources: dict[str, SubjectRoiSource] = {}
    for roi in ROI_ORDER:
        selection_key = f"{model}/{roi}"
        try:
            selection = reference_manifest["source_selection"][selection_key]
            artifact = reference_manifest["artifacts"][model][roi]
            selected_name = str(selection["selected"])
            artifact_name = str(artifact["source_file"])
            expected_sha = str(artifact["source_sha256"])
        except (KeyError, TypeError) as error:
            raise SelectivityAtlasValidationError(
                f"Reference manifest has no complete audited source record for {selection_key}"
            ) from error
        if selected_name != artifact_name:
            raise SelectivityAtlasValidationError(
                f"Reference manifest disagrees on the source file for {selection_key}: "
                f"{selected_name!r} vs {artifact_name!r}"
            )
        sources[roi] = load_subject1_csv(
            raw_dir / selected_name,
            model=model,
            roi=roi,
            expected_count=int(ROI_SPECS[roi]["responseCount"]),
            expected_image_ids=image_ids,
            expected_sha256=expected_sha,
        )

    orders = {source.image_ids for source in sources.values()}
    if len(orders) != 1:
        raise SelectivityAtlasValidationError(
            f"{model}: FFA, EBA, and PPA do not have identical image order"
        )
    return sources


def combine_model_sources(
    sources: Mapping[str, SubjectRoiSource],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Concatenate FFA/EBA/PPA while retaining both global and ROI indices."""
    missing = [roi for roi in ROI_ORDER if roi not in sources]
    if missing:
        raise SelectivityAtlasValidationError(f"Missing ROI sources: {', '.join(missing)}")
    image_orders = {sources[roi].image_ids for roi in ROI_ORDER}
    if len(image_orders) != 1:
        raise SelectivityAtlasValidationError(
            "FFA, EBA, and PPA image identifiers/order must be identical"
        )

    matrices: list[np.ndarray] = []
    roi_names: list[str] = []
    roi_indices: list[int] = []
    for roi in ROI_ORDER:
        matrix = np.asarray(sources[roi].responses, dtype=np.float64)
        expected_shape = (
            len(sources[roi].image_ids),
            int(ROI_SPECS[roi]["responseCount"]),
        )
        if matrix.shape != expected_shape:
            raise SelectivityAtlasValidationError(
                f"{roi.upper()} matrix shape {matrix.shape}; expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise SelectivityAtlasValidationError(
                f"{roi.upper()} Subject 1 responses contain non-finite values"
            )
        matrices.append(matrix)
        roi_names.extend([roi] * matrix.shape[1])
        roi_indices.extend(range(matrix.shape[1]))

    combined = np.concatenate(matrices, axis=1)
    if combined.shape != (len(next(iter(image_orders))), VOXEL_COUNT):
        raise SelectivityAtlasValidationError(
            f"Joint response matrix shape {combined.shape}; expected "
            f"({len(next(iter(image_orders)))}, {VOXEL_COUNT})"
        )
    return (
        combined,
        np.asarray(roi_names, dtype="<U3"),
        np.asarray(roi_indices, dtype=np.int32),
        next(iter(image_orders)),
    )


def selectivity_features(
    responses: np.ndarray,
    *,
    n_components: int = 35,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply mentor preprocessing: voxel z-score, transpose, PCA, L2 norm."""
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise SelectivityAtlasValidationError(
            "Response matrix must be a finite 2D images x voxels array"
        )
    profile_matrix = matrix.T
    if not 1 <= n_components <= min(profile_matrix.shape):
        raise SelectivityAtlasValidationError(
            f"PCA component count {n_components} is invalid for {profile_matrix.shape}"
        )

    means = matrix.mean(axis=0, keepdims=True)
    standard_deviations = matrix.std(axis=0, keepdims=True)
    z_profiles = (matrix - means) / (standard_deviations + 1e-9)
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=0,
    )
    features = pca.fit_transform(z_profiles.T)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    if not np.isfinite(features).all():
        raise SelectivityAtlasValidationError(
            "Selectivity preprocessing produced non-finite features"
        )
    return features, z_profiles, float(pca.explained_variance_ratio_.sum())


def embed_selectivity_3d(features: np.ndarray, config: AtlasConfig) -> np.ndarray:
    """Fit the display-only UMAP embedding, importing the optional dependency lazily."""
    cache_dir = Path(tempfile.gettempdir()) / "infoviz-selectivity-atlas-numba-cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import umap
    except ImportError as error:  # pragma: no cover - depends on build environment.
        raise SelectivityAtlasValidationError(
            "umap-learn is required to build browser coordinates; install "
            "scripts/data/requirements.txt"
        ) from error
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.umap_random_state,
        n_jobs=1,
    )
    embedding = np.asarray(reducer.fit_transform(features), dtype=np.float64)
    if embedding.shape != (features.shape[0], 3) or not np.isfinite(embedding).all():
        raise SelectivityAtlasValidationError(
            f"UMAP returned invalid embedding shape/data: {embedding.shape}"
        )
    return embedding


def _cluster_features(features: np.ndarray, config: AtlasConfig) -> tuple[np.ndarray, KMeans]:
    kmeans = KMeans(
        n_clusters=config.cluster_count,
        n_init=config.kmeans_n_init,
        random_state=config.kmeans_random_state,
    )
    labels = np.asarray(kmeans.fit_predict(features), dtype=np.int32)
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.arange(config.cluster_count)):
        raise SelectivityAtlasValidationError(
            f"K-means did not populate all {config.cluster_count} clusters"
        )
    return labels, kmeans


def evaluate_split_half_stability(
    responses: np.ndarray,
    *,
    config: AtlasConfig = AtlasConfig(),
    iterations: int = SPLIT_HALF_ITERATIONS,
    random_seed: int = SPLIT_HALF_RANDOM_SEED,
    threshold: float = SPLIT_HALF_ARI_THRESHOLD,
) -> dict[str, Any]:
    """Compare independently refit voxel labels across complementary image halves."""
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise SelectivityAtlasValidationError(
            "Split-half stability requires a finite 2D images x voxels matrix"
        )
    if iterations < 1:
        raise SelectivityAtlasValidationError("Split-half iterations must be positive")
    split_index = matrix.shape[0] // 2
    if min(split_index, matrix.shape[0] - split_index) < config.pca_components:
        raise SelectivityAtlasValidationError(
            "Each split half must contain at least as many images as PCA components"
        )

    rng = np.random.default_rng(random_seed)
    scores: list[float] = []
    for _ in range(iterations):
        permutation = rng.permutation(matrix.shape[0])
        first_half = matrix[permutation[:split_index], :]
        second_half = matrix[permutation[split_index:], :]
        first_features, _, _ = selectivity_features(
            first_half, n_components=config.pca_components
        )
        second_features, _, _ = selectivity_features(
            second_half, n_components=config.pca_components
        )
        first_labels, _ = _cluster_features(first_features, config)
        second_labels, _ = _cluster_features(second_features, config)
        scores.append(float(adjusted_rand_score(first_labels, second_labels)))

    mean_score = float(np.mean(scores))
    minimum_score = float(np.min(scores))
    return {
        "iterations": iterations,
        "randomSeed": random_seed,
        "adjustedRandIndices": scores,
        "meanAdjustedRandIndex": mean_score,
        "minAdjustedRandIndex": minimum_score,
        "threshold": float(threshold),
        "meanPassesThreshold": bool(mean_score >= threshold),
        "allPassThreshold": bool(all(score >= threshold for score in scores)),
    }


def fit_selectivity_atlas(
    responses: np.ndarray,
    *,
    config: AtlasConfig = AtlasConfig(),
    embedder: Callable[[np.ndarray, AtlasConfig], np.ndarray] = embed_selectivity_3d,
) -> AtlasFit:
    features, z_profiles, explained = selectivity_features(
        responses, n_components=config.pca_components
    )
    if not 2 <= config.cluster_count < features.shape[0]:
        raise SelectivityAtlasValidationError(
            f"Cluster count {config.cluster_count} must be between 2 and "
            f"{features.shape[0] - 1}"
        )
    labels, kmeans = _cluster_features(features, config)
    distances = np.linalg.norm(features - kmeans.cluster_centers_[labels], axis=1)
    samples = silhouette_samples(features, labels, metric="euclidean")
    overall_silhouette = silhouette_score(features, labels, metric="euclidean")
    embedding = np.asarray(embedder(features, config), dtype=np.float64)
    if embedding.shape != (features.shape[0], 3) or not np.isfinite(embedding).all():
        raise SelectivityAtlasValidationError(
            f"Display embedder returned invalid shape/data: {embedding.shape}"
        )
    return AtlasFit(
        z_profiles=z_profiles,
        selectivity_features=features,
        pca_explained_variance_ratio_sum=explained,
        labels=labels,
        embedding=embedding,
        distances_to_centroid=np.asarray(distances, dtype=np.float64),
        silhouette_by_voxel=np.asarray(samples, dtype=np.float64),
        silhouette=float(overall_silhouette),
        split_half_stability=evaluate_split_half_stability(
            responses,
            config=config,
        ),
    )


def _ranked_images(
    profile: np.ndarray,
    image_ids: Sequence[str],
    *,
    largest: bool,
    count: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-profile if largest else profile, kind="stable")[:count]
    return [
        {"imageId": str(image_ids[index]), "meanZ": float(profile[index])}
        for index in order
    ]


def build_summary(
    *,
    model: str,
    image_ids: Sequence[str],
    roi_names: np.ndarray,
    fit: AtlasFit,
    source_files: Sequence[Mapping[str, Any]],
    config: AtlasConfig,
    top_image_count: int = TOP_IMAGE_COUNT,
) -> dict[str, Any]:
    if fit.z_profiles.shape != (len(image_ids), len(roi_names)):
        raise SelectivityAtlasValidationError(
            "Z-profile dimensions do not match image and voxel metadata"
        )
    clusters: list[dict[str, Any]] = []
    for label in range(config.cluster_count):
        mask = fit.labels == label
        if not np.any(mask):
            raise SelectivityAtlasValidationError(f"Cluster {label} is empty")
        mean_profile = fit.z_profiles[:, mask].mean(axis=1)
        clusters.append(
            {
                "label": label,
                "size": int(mask.sum()),
                "roiCounts": {
                    roi: int(np.sum(roi_names[mask] == roi)) for roi in ROI_ORDER
                },
                "meanSilhouette": float(fit.silhouette_by_voxel[mask].mean()),
                "preferredImages": _ranked_images(
                    mean_profile,
                    image_ids,
                    largest=True,
                    count=top_image_count,
                ),
                "suppressedImages": _ranked_images(
                    mean_profile,
                    image_ids,
                    largest=False,
                    count=top_image_count,
                ),
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "dataset": DATASET,
        "subject": "s1",
        **provenance_metadata(),
        "model": model,
        "imageCount": len(image_ids),
        "voxelCount": len(roi_names),
        "clusterCount": config.cluster_count,
        "silhouette": fit.silhouette,
        "pcaExplainedVarianceRatioSum": fit.pca_explained_variance_ratio_sum,
        "method": method_metadata(config),
        "k14StabilityCaveat": K14_STABILITY_CAVEAT,
        "splitHalfStability": dict(fit.split_half_stability),
        "sourceFiles": [dict(record) for record in source_files],
        "clusters": clusters,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        raise SelectivityAtlasValidationError("Cannot serialize a non-finite point value")
    return format(float(value), ".9g")


def write_points_csv(
    path: Path,
    *,
    roi_names: np.ndarray,
    roi_indices: np.ndarray,
    fit: AtlasFit,
) -> None:
    point_count = len(fit.labels)
    if len(roi_names) != point_count or len(roi_indices) != point_count:
        raise SelectivityAtlasValidationError(
            "Point metadata lengths do not match fitted voxel count"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "voxel_index",
                "roi",
                "roi_voxel_index",
                "cluster_label",
                "umap_x",
                "umap_y",
                "umap_z",
                "distance_to_centroid",
            )
        )
        for index in range(point_count):
            writer.writerow(
                (
                    index,
                    str(roi_names[index]),
                    int(roi_indices[index]),
                    int(fit.labels[index]),
                    _format_float(fit.embedding[index, 0]),
                    _format_float(fit.embedding[index, 1]),
                    _format_float(fit.embedding[index, 2]),
                    _format_float(fit.distances_to_centroid[index]),
                )
            )


def _roi_manifest() -> dict[str, dict[str, Any]]:
    offset = 0
    result: dict[str, dict[str, Any]] = {}
    for roi in ROI_ORDER:
        count = int(ROI_SPECS[roi]["responseCount"])
        result[roi] = {
            "offset": offset,
            "count": count,
        }
        offset += count
    return result


def _install_staged_directory(staging: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
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


def build(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    reference_manifest_path: Path = DEFAULT_REFERENCE_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models: Sequence[str] | None = None,
    config: AtlasConfig = AtlasConfig(),
) -> dict[str, Any]:
    reference = load_reference_manifest(reference_manifest_path)
    supported_models = tuple(str(model) for model in reference["supported_models"])
    selected_models = tuple(models) if models else supported_models
    unknown = sorted(set(selected_models) - set(supported_models))
    if unknown:
        raise SelectivityAtlasValidationError(
            f"Models are not in the audited reference manifest: {', '.join(unknown)}"
        )
    if len(selected_models) != len(set(selected_models)):
        raise SelectivityAtlasValidationError("Requested model list contains duplicates")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-build-", dir=output_dir.parent)
    )
    model_records: dict[str, dict[str, Any]] = {}
    try:
        for model in selected_models:
            sources = load_model_sources(
                model=model,
                raw_dir=raw_dir,
                reference_manifest=reference,
            )
            responses, roi_names, roi_indices, image_ids = combine_model_sources(sources)
            fit = fit_selectivity_atlas(responses, config=config)
            source_files = [
                {
                    "roi": roi,
                    "fileName": sources[roi].path.name,
                    "sha256": sources[roi].sha256,
                }
                for roi in ROI_ORDER
            ]
            model_dir = staging / model
            points_path = model_dir / "points.csv"
            summary_path = model_dir / "summary.json"
            write_points_csv(
                points_path,
                roi_names=roi_names,
                roi_indices=roi_indices,
                fit=fit,
            )
            write_json(
                summary_path,
                build_summary(
                    model=model,
                    image_ids=image_ids,
                    roi_names=roi_names,
                    fit=fit,
                    source_files=source_files,
                    config=config,
                ),
            )
            model_records[model] = {
                "pointsUrl": f"{model}/points.csv",
                "summaryUrl": f"{model}/summary.json",
                "pointsSha256": sha256_file(points_path),
                "summarySha256": sha256_file(summary_path),
                "pointsBytes": points_path.stat().st_size,
                "summaryBytes": summary_path.stat().st_size,
                "pointCount": len(fit.labels),
            }
            print(
                f"Built {model}: {len(fit.labels)} voxels, "
                f"silhouette={fit.silhouette:.4f}",
                flush=True,
            )

        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "artifact": "voxel-selectivity-atlas",
            "dataset": DATASET,
            "subject": "s1",
            **provenance_metadata(),
            "imageCount": IMAGE_COUNT,
            "voxelCount": VOXEL_COUNT,
            "clusterCount": config.cluster_count,
            "roiOrder": list(ROI_ORDER),
            "rois": _roi_manifest(),
            "method": method_metadata(config),
            "k14StabilityCaveat": K14_STABILITY_CAVEAT,
            "supportedModels": list(selected_models),
            "referenceVersion": reference.get("reference_version"),
            "imageIdsSha256": sha256_json(list(canonical_image_ids())),
            "models": model_records,
        }
        write_json(staging / "manifest.json", manifest)
        _install_staged_directory(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--append-baseline-atlas", type=Path, help="Stage new audited models while copying this pinned baseline atlas unchanged.")
    mode.add_argument("--standardize-from-baseline", type=Path, help="Refit the exact full model roster from verified FP32 predictions into a fresh staging directory.")
    parser.add_argument("--baseline-artifact-root", type=Path, default=REPO_ROOT / "artifacts/reference")
    parser.add_argument("--audit-path", type=Path, default=REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "public/images")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional audited model subset (default: build every supported model)",
    )
    return parser


def append_selectivity_atlas_assets(
    *,
    raw_root: Path,
    artifact_root: Path,
    baseline_artifact_root: Path,
    audit_path: Path,
    baseline_atlas_root: Path,
    output_dir: Path,
    image_root: Path = REPO_ROOT / "public/images",
    config: AtlasConfig = AtlasConfig(),
) -> dict[str, Any]:
    """Stage only newly audited fits and copy every baseline child unchanged.

    The full pooled order and checkpoint lineage must pass the same independent
    audit as the anatomical export. UMAP is still display-only; each new model
    is fit independently using the existing fixed exploratory k=14 method.
    This stages assets only and never promotes frontend manifest digest pins.
    """
    try:
        from .build_brain_reference_artifacts import validate_spatial_expansion
    except ImportError:  # pragma: no cover - direct script execution
        from build_brain_reference_artifacts import validate_spatial_expansion
    if output_dir.exists():
        raise SelectivityAtlasValidationError("Expansion output must be a new staging directory.")
    if config != AtlasConfig():
        raise SelectivityAtlasValidationError("An append must retain the baseline selectivity method exactly.")
    if sha256_file(baseline_atlas_root / "manifest.json") != BASELINE_ATLAS_MANIFEST_SHA256:
        raise SelectivityAtlasValidationError("Baseline atlas manifest does not match the reviewed digest.")
    baseline = json.loads((baseline_atlas_root / "manifest.json").read_text(encoding="utf-8"))
    reference, contract, matrices = validate_spatial_expansion(
        artifact_root=artifact_root, raw_root=raw_root, baseline_artifact_root=baseline_artifact_root,
        audit_path=audit_path, image_root=image_root,
    )
    baseline_models = set(reference["supported_models"]) - set(matrices)
    if (
        set(baseline["supportedModels"]) != baseline_models
        or set(baseline["models"]) != baseline_models
        or baseline["method"] != method_metadata(config)
        or baseline["referenceVersion"] != contract["baseline_reference_version"]
    ):
        raise SelectivityAtlasValidationError("Baseline atlas coverage or method is incompatible with the spatial audit.")
    # Verify all old children before creating the output directory.
    baseline_files: list[tuple[Path, Path]] = []
    for model in sorted(baseline_models):
        record = baseline["models"][model]
        for kind, filename in (("points", "points.csv"), ("summary", "summary.json")):
            relative = f"{model}/{filename}"
            if record[f"{kind}Url"] != relative:
                raise SelectivityAtlasValidationError("Unsafe baseline atlas child path.")
            path = baseline_atlas_root / relative
            if sha256_file(path) != record[f"{kind}Sha256"]:
                raise SelectivityAtlasValidationError(f"{model}: baseline atlas {kind} changed.")
            baseline_files.append((path, Path(relative)))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-append-", dir=output_dir.parent))
    records = dict(baseline["models"])
    try:
        for source, relative in baseline_files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for model in sorted(matrices):
            sources = load_model_sources(
                model=model, raw_dir=raw_root / model, reference_manifest=reference,
            )
            responses, roi_names, roi_indices, image_ids = combine_model_sources(sources)
            fit = fit_selectivity_atlas(responses, config=config)
            model_dir = staging / model
            points_path = model_dir / "points.csv"
            summary_path = model_dir / "summary.json"
            write_points_csv(points_path, roi_names=roi_names, roi_indices=roi_indices, fit=fit)
            write_json(summary_path, build_summary(
                model=model, image_ids=image_ids, roi_names=roi_names, fit=fit,
                source_files=[{"roi": roi, "fileName": sources[roi].path.name, "sha256": sources[roi].sha256} for roi in ROI_ORDER],
                config=config,
            ))
            records[model] = {
                "pointsUrl": f"{model}/points.csv", "summaryUrl": f"{model}/summary.json",
                "pointsSha256": sha256_file(points_path), "summarySha256": sha256_file(summary_path),
                "pointsBytes": points_path.stat().st_size, "summaryBytes": summary_path.stat().st_size,
                "pointCount": len(fit.labels),
            }
            print(f"Appended {model}: {len(fit.labels)} voxels, silhouette={fit.silhouette:.4f}", flush=True)
        manifest = {
            **baseline,
            "supportedModels": sorted(reference["supported_models"]),
            "referenceVersion": reference["reference_version"],
            "models": records,
        }
        write_json(staging / "manifest.json", manifest)
        # A new directory only: never move or remove an existing user bundle.
        if output_dir.exists():
            raise SelectivityAtlasValidationError("Expansion output appeared during generation; nothing was replaced.")
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def standardize_selectivity_atlas_assets(
    *, raw_root: Path, artifact_root: Path, baseline_root: Path, audit_path: Path,
    output_dir: Path, image_root: Path = REPO_ROOT / "public/images", config: AtlasConfig = AtlasConfig(),
) -> dict[str, Any]:
    """Refit every known model without altering the method or unrelated artifacts."""
    try:
        from .spatial_standardization import PARENT_ATLAS_MANIFEST_SHA256, checked_document, validate_spatial_standardization
    except ImportError:  # pragma: no cover - direct script execution
        from spatial_standardization import PARENT_ATLAS_MANIFEST_SHA256, checked_document, validate_spatial_standardization
    if output_dir.exists():
        raise SelectivityAtlasValidationError("Standardization output must be a new staging directory.")
    if config != AtlasConfig():
        raise SelectivityAtlasValidationError("Standardization must retain the reviewed selectivity method exactly.")
    reference, contract, matrices = validate_spatial_standardization(
        artifact_root=artifact_root, raw_root=raw_root, baseline_root=baseline_root,
        audit_path=audit_path, image_root=image_root,
    )
    parent_root = baseline_root / "public/data/voxel_selectivity_atlas"
    parent = checked_document(parent_root / "manifest.json", PARENT_ATLAS_MANIFEST_SHA256)
    if parent["method"] != method_metadata(config):
        raise SelectivityAtlasValidationError("The reviewed parent selectivity method is incompatible.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-standardize-", dir=output_dir.parent))
    records = {}
    try:
        shutil.copytree(parent_root, staging, dirs_exist_ok=True)
        for model in sorted(matrices):
            sources = load_model_sources(model=model, raw_dir=raw_root / model, reference_manifest=reference)
            responses, roi_names, roi_indices, image_ids = combine_model_sources(sources)
            # The clustering input must be exactly the same raw Subject-1
            # prefix as the independent anatomical export, just reordered.
            expected = np.concatenate([matrices[model][roi] for roi in ROI_ORDER], axis=1)
            if not np.array_equal(responses.astype(np.float32), expected):
                raise SelectivityAtlasValidationError(f"{model}: atlas source differs from audited raw S1 responses.")
            # Start from exact audited float32 prediction bits, not the tiny
            # decimal-roundtrip differences in a float64 CSV parser. Analysis
            # precision and the fixed clustering method are otherwise unchanged.
            fit = fit_selectivity_atlas(expected.astype(np.float64), config=config)
            points_path = staging / model / "points.csv"
            summary_path = staging / model / "summary.json"
            write_points_csv(points_path, roi_names=roi_names, roi_indices=roi_indices, fit=fit)
            write_json(summary_path, build_summary(
                model=model, image_ids=image_ids, roi_names=roi_names, fit=fit,
                source_files=[{"roi": roi, "fileName": sources[roi].path.name, "sha256": sources[roi].sha256} for roi in ROI_ORDER],
                config=config,
            ))
            records[model] = {
                "pointsUrl": f"{model}/points.csv", "summaryUrl": f"{model}/summary.json",
                "pointsSha256": sha256_file(points_path), "summarySha256": sha256_file(summary_path),
                "pointsBytes": points_path.stat().st_size, "summaryBytes": summary_path.stat().st_size,
                "pointCount": len(fit.labels),
            }
            print(f"Standardized {model}: {len(fit.labels)} voxels, silhouette={fit.silhouette:.4f}", flush=True)
        manifest = {**parent, "supportedModels": sorted(matrices), "referenceVersion": contract["reference_version"], "models": records}
        write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise SelectivityAtlasValidationError("Standardization output appeared during generation; nothing was replaced.")
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.standardize_from_baseline:
            if args.models:
                raise SelectivityAtlasValidationError("A standardization must cover all models, not a subset.")
            manifest = standardize_selectivity_atlas_assets(
                raw_root=args.raw_dir, artifact_root=args.reference_manifest.parent,
                baseline_root=args.standardize_from_baseline, audit_path=args.audit_path,
                output_dir=args.output_dir, image_root=args.image_root,
            )
        elif args.append_baseline_atlas:
            if args.models:
                raise SelectivityAtlasValidationError("An append must cover the exact expanded reference roster, not a separate model subset.")
            manifest = append_selectivity_atlas_assets(
                raw_root=args.raw_dir, artifact_root=args.reference_manifest.parent,
                baseline_artifact_root=args.baseline_artifact_root, audit_path=args.audit_path,
                baseline_atlas_root=args.append_baseline_atlas, output_dir=args.output_dir, image_root=args.image_root,
            )
        else:
            manifest = build(
                raw_dir=args.raw_dir,
                reference_manifest_path=args.reference_manifest,
                output_dir=args.output_dir,
                models=args.models,
            )
    except SelectivityAtlasValidationError as error:
        raise SystemExit(f"Selectivity atlas build failed: {error}") from error
    print(
        f"{'Staged' if args.append_baseline_atlas or args.standardize_from_baseline else 'Installed'} {len(manifest['supportedModels'])} model atlas artifacts at "
        f"{args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
