#!/usr/bin/env python3
"""Build local-browser NSD Subject 1 cortical projection artifacts.

The NSD-1000 model response order is reconstructed from the training-era
Subject 1 cortical mask, fLoc labels, and t-values.  The selected func1pt8
voxels are then projected onto the exact Subject 1 surface vertex order with
the cached pycortex ``linenn`` mapper.

This script intentionally does *not* use ``surfacevoxels_layerB2.mgz`` from
the mentor archive.  That file is a three-frame NSD sampling test pattern, not
a volume-linear-index lookup.  It also intentionally uses the model's t > 7
selection rather than the archive's illustrative t > 3.3 exporter threshold.

The public output contains axis-rotated, full-resolution pial and inflated
surface meshes plus normalized sparse mapping weights. Raw masks, t-values,
response tensors, and transforms are not copied into the web application.
The source archive supplied no redistribution terms, so generated meshes must
remain local unless the anatomy owner grants deployment/redistribution rights.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
RESPONSE_SPACE = "nsd_1000_subject1_floc_t7_v1"
VOLUME_SHAPE_ZYX = (81, 104, 83)
VOLUME_SHAPE_XYZ = (83, 104, 81)
THRESHOLD = 7.0

ROI_SPECS = {
    "ffa": {
        "label": "FFA",
        "domain": "faces",
        "label_values": (2, 3),
        "response_count": 156,
        "color": "#C95C54",
    },
    "eba": {
        "label": "EBA",
        "domain": "bodies",
        "label_values": (1,),
        "response_count": 518,
        "color": "#3C8D7C",
    },
    "ppa": {
        "label": "PPA",
        "domain": "places",
        "label_values": (2,),
        "response_count": 170,
        "color": "#4C72B0",
    },
}

# These hashes pin the selected voxel identity *and order* established in the
# repository audit.  Coordinates are little-endian int16 (z, y, x), C-order.
EXPECTED_COORDINATE_SHA256 = {
    "ffa": "ce13c8bcd14cc54865b90cc4d9546559afd92269ad2bdc9a66df9aa327ef8a88",
    "eba": "440e3d36e725f44df7a4bfc6aeab9cc3fc180853adde80843439393698c36fbe",
    "ppa": "e0b164fe5a54d564b09e4dc25b4d456bb966933870e3dbb9aebd93d73657282e",
}

DEFAULT_NSD_DATA_ROOT = Path(
    "/research/mainak/repos/nesim/training/nsd/nsd-data/nsd"
)
DEFAULT_BODY_LABELS = Path(
    "/research/mainak/repos/mindeyev2_lesion/nsd_processed_om/masks/subj1/"
    "roi_1d_mask_subj01_floc-bodies.npy"
)
DEFAULT_MAPPER = Path(
    "/research/mainak/repos/pycortex_viz/subj01/cache/"
    "func1pt8_to_anat0pt8_autoFSbbr_linenn.npz"
)
DEFAULT_TENSOR_ROOT = Path("/research/kushal/topo-dynamics/data/nsd")
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "artifacts/neuroimaging/nsd_subj01/source/nsd_subj01_froi_source.zip"
)
DEFAULT_PUBLIC_OUT = REPO_ROOT / "public/data/brain/nsd-subj01"
DEFAULT_DERIVED_OUT = (
    REPO_ROOT / "artifacts/neuroimaging/nsd_subj01/derived"
)


class BrainArtifactValidationError(ValueError):
    """Raised when a source violates the audited Subject 1 contract."""


@dataclass(frozen=True)
class RoiSelection:
    name: str
    coordinates_zyx: np.ndarray
    linear_indices_xyz: np.ndarray


@dataclass(frozen=True)
class HemisphereMapping:
    response_indices: np.ndarray
    vertex_indices: np.ndarray
    raw_weights: np.ndarray
    response_normalized_weights: np.ndarray
    vertex_normalized_weights: np.ndarray


@dataclass(frozen=True)
class RoiProjection:
    selection: RoiSelection
    mappings: Mapping[str, HemisphereMapping]
    hemispheres: tuple[str, ...]
    unmapped_response_indices: tuple[int, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise BrainArtifactValidationError(f"Missing {description}: {path}")
    return path


def _as_validated_1d(path: Path, expected_length: int, description: str) -> np.ndarray:
    values = np.load(_require_file(path, description), allow_pickle=False)
    values = np.asarray(values).squeeze()
    if values.shape != (expected_length,):
        raise BrainArtifactValidationError(
            f"{description} must have shape ({expected_length},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise BrainArtifactValidationError(f"{description} contains non-finite values")
    return values


def reconstruct_roi_selection(
    *,
    name: str,
    cortical_mask_zyx: np.ndarray,
    roi_labels_1d: np.ndarray,
    t_values_1d: np.ndarray,
    label_values: Sequence[int],
    threshold: float = THRESHOLD,
    expected_count: int | None = None,
    expected_coordinate_sha256: str | None = None,
) -> RoiSelection:
    """Rebuild model-column coordinates in NumPy boolean-index order."""
    mask = np.asarray(cortical_mask_zyx)
    if mask.shape != VOLUME_SHAPE_ZYX:
        raise BrainArtifactValidationError(
            f"Cortical mask must have z/y/x shape {VOLUME_SHAPE_ZYX}, got {mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise BrainArtifactValidationError(f"Cortical mask must be boolean, got {mask.dtype}")

    n_cortical = int(mask.sum())
    labels = np.asarray(roi_labels_1d).squeeze()
    t_values = np.asarray(t_values_1d).squeeze()
    for values, description in ((labels, "ROI labels"), (t_values, "t-values")):
        if values.shape != (n_cortical,):
            raise BrainArtifactValidationError(
                f"{name} {description} must have shape ({n_cortical},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise BrainArtifactValidationError(f"{name} {description} contains non-finite values")

    selected_1d = np.isin(labels, tuple(label_values)) & (t_values > threshold)
    selected_volume = np.zeros(mask.shape, dtype=np.bool_)
    selected_volume[mask] = selected_1d
    coordinates_zyx = np.ascontiguousarray(
        np.argwhere(selected_volume), dtype="<i2"
    )

    if expected_count is not None and len(coordinates_zyx) != expected_count:
        raise BrainArtifactValidationError(
            f"{name.upper()} t>{threshold:g} selection has {len(coordinates_zyx)} voxels; "
            f"expected {expected_count}"
        )

    coordinate_hash = sha256_bytes(coordinates_zyx.tobytes(order="C"))
    if expected_coordinate_sha256 is not None and coordinate_hash != expected_coordinate_sha256:
        raise BrainArtifactValidationError(
            f"{name.upper()} ordered coordinate hash {coordinate_hash} does not match "
            f"the audited contract {expected_coordinate_sha256}"
        )

    z, y, x = coordinates_zyx.T
    linear_indices_xyz = np.ascontiguousarray(
        np.ravel_multi_index((x, y, z), VOLUME_SHAPE_XYZ, order="C"),
        dtype="<i4",
    )
    if len(np.unique(linear_indices_xyz)) != len(linear_indices_xyz):
        raise BrainArtifactValidationError(f"{name.upper()} contains duplicate volume voxels")

    return RoiSelection(
        name=name,
        coordinates_zyx=coordinates_zyx,
        linear_indices_xyz=linear_indices_xyz,
    )


def load_roi_selections(
    nsd_data_root: Path,
    body_labels_path: Path,
) -> tuple[dict[str, RoiSelection], dict[str, Path]]:
    mask_path = nsd_data_root / "voxels_masks/subj1/cortical_mask_subj01.npy"
    cortical_mask = np.load(_require_file(mask_path, "Subject 1 cortical mask"), allow_pickle=False)
    if cortical_mask.shape != VOLUME_SHAPE_ZYX or cortical_mask.dtype != np.bool_:
        raise BrainArtifactValidationError(
            f"Subject 1 cortical mask must be bool {VOLUME_SHAPE_ZYX}, got "
            f"{cortical_mask.dtype} {cortical_mask.shape}"
        )
    n_cortical = int(cortical_mask.sum())

    source_paths: dict[str, Path] = {"cortical_mask": mask_path}
    labels_by_roi: dict[str, np.ndarray] = {}
    t_values_by_roi: dict[str, np.ndarray] = {}
    for roi, stem in (("ffa", "faces"), ("ppa", "places")):
        label_path = (
            nsd_data_root
            / f"voxels_masks/subj1/roi_1d_mask_subj01_floc-{stem}.npy"
        )
        labels_by_roi[roi] = _as_validated_1d(
            label_path, n_cortical, f"{roi.upper()} training-era labels"
        )
        source_paths[f"{roi}_labels"] = label_path

    labels_by_roi["eba"] = _as_validated_1d(
        body_labels_path, n_cortical, "EBA training-compatible body labels"
    )
    source_paths["eba_labels"] = body_labels_path

    for roi, stem in (("ffa", "faces"), ("eba", "bodies"), ("ppa", "places")):
        t_path = nsd_data_root / f"tvals/subj1_{stem}.npy"
        t_values_by_roi[roi] = _as_validated_1d(
            t_path, n_cortical, f"{roi.upper()} training-era t-values"
        )
        source_paths[f"{roi}_t_values"] = t_path

    selections = {
        roi: reconstruct_roi_selection(
            name=roi,
            cortical_mask_zyx=cortical_mask,
            roi_labels_1d=labels_by_roi[roi],
            t_values_1d=t_values_by_roi[roi],
            label_values=spec["label_values"],
            expected_count=spec["response_count"],
            expected_coordinate_sha256=EXPECTED_COORDINATE_SHA256[roi],
        )
        for roi, spec in ROI_SPECS.items()
    }
    return selections, source_paths


def _load_csr_mapper(archive: Mapping[str, np.ndarray], prefix: str) -> sparse.csr_matrix:
    required = ("data", "indices", "indptr", "shape")
    missing = [f"{prefix}_{suffix}" for suffix in required if f"{prefix}_{suffix}" not in archive]
    if missing:
        raise BrainArtifactValidationError(f"Mapper is missing arrays: {', '.join(missing)}")
    matrix = sparse.csr_matrix(
        (
            np.asarray(archive[f"{prefix}_data"], dtype=np.float64),
            np.asarray(archive[f"{prefix}_indices"], dtype=np.int32),
            np.asarray(archive[f"{prefix}_indptr"], dtype=np.int32),
        ),
        shape=tuple(int(value) for value in archive[f"{prefix}_shape"]),
    )
    if matrix.shape[1] != int(np.prod(VOLUME_SHAPE_XYZ)):
        raise BrainArtifactValidationError(
            f"{prefix} mapper volume width is {matrix.shape[1]}, expected "
            f"{int(np.prod(VOLUME_SHAPE_XYZ))}"
        )
    if matrix.nnz == 0 or not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise BrainArtifactValidationError(f"{prefix} mapper weights are invalid")
    return matrix


def project_selection(
    selection: RoiSelection,
    mappers: Mapping[str, sparse.csr_matrix],
) -> RoiProjection:
    """Slice mapper columns and derive rendering plus forensic normalizations.

    Browser rendering uses per-surface-vertex (mapper-row) normalization after
    filtering to one ROI.  Per-response/column normalization is retained only
    in the ignored full artifact to audit which vertices each response reaches.
    """
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    response_sums = np.zeros(len(selection.coordinates_zyx), dtype=np.float64)
    hemisphere_sums: dict[str, np.ndarray] = {}

    for hemi in ("lh", "rh"):
        if hemi not in mappers:
            raise BrainArtifactValidationError(f"Missing {hemi} surface mapper")
        columns = mappers[hemi][:, selection.linear_indices_xyz].tocsc()
        counts = np.diff(columns.indptr)
        response_indices = np.repeat(np.arange(columns.shape[1], dtype=np.int32), counts)
        vertex_indices = np.asarray(columns.indices, dtype=np.int32)
        weights = np.asarray(columns.data, dtype=np.float64)
        raw[hemi] = (response_indices, vertex_indices, weights)
        sums = np.bincount(
            response_indices, weights=weights, minlength=columns.shape[1]
        ).astype(np.float64)
        hemisphere_sums[hemi] = sums
        response_sums += sums

    mappings: dict[str, HemisphereMapping] = {}
    for hemi in ("lh", "rh"):
        response_indices, vertex_indices, weights = raw[hemi]
        if len(weights):
            response_normalized = weights / response_sums[response_indices]
            vertex_sums = np.bincount(
                vertex_indices,
                weights=weights,
                minlength=mappers[hemi].shape[0],
            ).astype(np.float64)
            vertex_normalized = weights / vertex_sums[vertex_indices]
        else:
            response_normalized = weights.copy()
            vertex_normalized = weights.copy()
        mappings[hemi] = HemisphereMapping(
            response_indices=np.ascontiguousarray(response_indices, dtype="<i4"),
            vertex_indices=np.ascontiguousarray(vertex_indices, dtype="<i4"),
            raw_weights=np.ascontiguousarray(weights, dtype="<f8"),
            response_normalized_weights=np.ascontiguousarray(
                response_normalized, dtype="<f4"
            ),
            vertex_normalized_weights=np.ascontiguousarray(
                vertex_normalized, dtype="<f4"
            ),
        )

    unmapped = tuple(int(index) for index in np.flatnonzero(response_sums == 0))
    hemispheres: list[str] = []
    for index in range(len(response_sums)):
        left = hemisphere_sums["lh"][index] > 0
        right = hemisphere_sums["rh"][index] > 0
        if left and right:
            hemispheres.append("bilateral")
        elif left:
            hemispheres.append("lh")
        elif right:
            hemispheres.append("rh")
        else:
            hemispheres.append("unmapped")

    mapped = response_sums > 0
    normalized_sums = np.zeros_like(response_sums)
    for mapping in mappings.values():
        normalized_sums += np.bincount(
            mapping.response_indices,
            weights=mapping.response_normalized_weights.astype(np.float64),
            minlength=len(response_sums),
        )
    if not np.allclose(normalized_sums[mapped], 1.0, atol=2e-6, rtol=0):
        raise BrainArtifactValidationError(
            f"{selection.name.upper()} forensic response-normalized mapper columns "
            "do not sum to one"
        )
    if np.any(normalized_sums[~mapped] != 0):
        raise BrainArtifactValidationError(
            f"{selection.name.upper()} unmapped responses unexpectedly have surface weights"
        )

    for hemi, mapping in mappings.items():
        vertex_sums = np.bincount(
            mapping.vertex_indices,
            weights=mapping.vertex_normalized_weights.astype(np.float64),
            minlength=mappers[hemi].shape[0],
        )
        used = vertex_sums > 0
        if not np.allclose(vertex_sums[used], 1.0, atol=2e-6, rtol=0):
            raise BrainArtifactValidationError(
                f"{selection.name.upper()} {hemi} vertex-normalized weights do not sum to one"
            )

    return RoiProjection(
        selection=selection,
        mappings=mappings,
        hemispheres=tuple(hemispheres),
        unmapped_response_indices=unmapped,
    )


def load_projections(
    selections: Mapping[str, RoiSelection], mapper_path: Path
) -> tuple[dict[str, RoiProjection], dict[str, int]]:
    with np.load(_require_file(mapper_path, "pycortex linenn mapper"), allow_pickle=False) as data:
        if "shape" not in data or tuple(int(v) for v in data["shape"]) != VOLUME_SHAPE_XYZ:
            raise BrainArtifactValidationError(
                f"Mapper must declare volume shape {VOLUME_SHAPE_XYZ}"
            )
        mappers = {
            "lh": _load_csr_mapper(data, "left"),
            "rh": _load_csr_mapper(data, "right"),
        }
    projections = {
        roi: project_selection(selection, mappers)
        for roi, selection in selections.items()
    }
    vertex_counts = {hemi: int(mapper.shape[0]) for hemi, mapper in mappers.items()}
    return projections, vertex_counts


def validate_training_tensors(tensor_root: Path, selections: Mapping[str, RoiSelection]) -> dict[str, Path]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by CLI environments
        raise BrainArtifactValidationError(
            "PyTorch is required to validate the NSD training tensor widths"
        ) from error

    paths: dict[str, Path] = {}
    for roi, selection in selections.items():
        path = _require_file(
            tensor_root / f"s1_{roi.upper()}_t7.pt",
            f"Subject 1 {roi.upper()} t7 training tensor",
        )
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != 2 or shape[1] != len(selection.coordinates_zyx):
            raise BrainArtifactValidationError(
                f"{path.name} shape {shape} does not match {len(selection.coordinates_zyx)} "
                f"ordered {roi.upper()} responses"
            )
        if shape[0] != 1000:
            raise BrainArtifactValidationError(
                f"{path.name} has {shape[0]} images; expected NSD-1000"
            )
        paths[f"{roi}_training_tensor"] = path
    return paths


def extract_surface_files(archive_path: Path, output_dir: Path) -> tuple[dict[str, dict[str, Path]], dict[str, str]]:
    """Extract only four surface files from the mentor archive."""
    members_by_suffix: dict[str, str] = {}
    member_hashes: dict[str, str] = {}
    surfaces: dict[str, dict[str, Path]] = {"inflated": {}, "pial": {}}
    with zipfile.ZipFile(_require_file(archive_path, "mentor surface archive")) as archive:
        names = archive.namelist()
        for variant in ("inflated", "pial"):
            for hemi in ("lh", "rh"):
                suffix = f"/web/{hemi}.{variant}"
                matches = [name for name in names if name.endswith(suffix)]
                if len(matches) != 1:
                    raise BrainArtifactValidationError(
                        f"Archive must contain exactly one *{suffix}; found {len(matches)}"
                    )
                member = matches[0]
                payload = archive.read(member)
                path = output_dir / f"{hemi}.{variant}"
                path.write_bytes(payload)
                members_by_suffix[f"{variant}_{hemi}"] = member
                member_hashes[f"{variant}_{hemi}"] = sha256_bytes(payload)
                surfaces[variant][hemi] = path
    return surfaces, member_hashes


def export_surface_glb(
    surface_paths: Mapping[str, Path],
    output_path: Path,
    expected_vertex_counts: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    try:
        import nibabel.freesurfer as fs
        import trimesh
    except ImportError as error:  # pragma: no cover - exercised by CLI environments
        raise BrainArtifactValidationError(
            "nibabel and trimesh are required to build cortical GLBs; install "
            "scripts/data/requirements.txt"
        ) from error

    scene = trimesh.Scene()
    metadata: dict[str, dict[str, int]] = {}
    for hemi in ("lh", "rh"):
        coordinates, faces = fs.read_geometry(str(surface_paths[hemi]))
        if coordinates.shape != (expected_vertex_counts[hemi], 3):
            raise BrainArtifactValidationError(
                f"{surface_paths[hemi].name} has {len(coordinates)} vertices; mapper expects "
                f"{expected_vertex_counts[hemi]}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise BrainArtifactValidationError(
                f"{surface_paths[hemi].name} is not a triangular surface"
            )
        if int(faces.min()) < 0 or int(faces.max()) >= len(coordinates):
            raise BrainArtifactValidationError(
                f"{surface_paths[hemi].name} contains out-of-bounds faces"
            )
        used_vertices = np.unique(faces)
        if len(used_vertices) != len(coordinates):
            raise BrainArtifactValidationError(
                f"{surface_paths[hemi].name} has {len(coordinates) - len(used_vertices)} "
                "vertices outside the mesh topology"
            )

        # FreeSurfer RAS (x, y, z) -> glTF right-handed, Y-up (x, z, -y).
        gltf_coordinates = np.ascontiguousarray(
            np.column_stack((coordinates[:, 0], coordinates[:, 2], -coordinates[:, 1])),
            dtype=np.float32,
        )
        mesh = trimesh.Trimesh(
            vertices=gltf_coordinates,
            faces=np.ascontiguousarray(faces, dtype=np.uint32),
            process=False,
        )
        scene.add_geometry(mesh, geom_name=hemi, node_name=hemi)
        metadata[hemi] = {
            "vertexCount": int(len(coordinates)),
            "faceCount": int(len(faces)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=False)
    output_path.write_bytes(payload)
    return metadata


def projection_document(projections: Mapping[str, RoiProjection]) -> dict[str, Any]:
    rois: dict[str, Any] = {}
    for roi, projection in projections.items():
        selection = projection.selection
        rois[roi] = {
            "responseCount": int(len(selection.coordinates_zyx)),
            "mappedResponseCount": int(
                len(selection.coordinates_zyx)
                - len(projection.unmapped_response_indices)
            ),
            "volumeCoordinatesZyx": selection.coordinates_zyx.astype(int).tolist(),
            "volumeLinearIndicesXyz": selection.linear_indices_xyz.astype(int).tolist(),
            "hemispheres": list(projection.hemispheres),
            "unmappedResponseIndices": list(projection.unmapped_response_indices),
            "mappings": {
                hemi: {
                    "responseIndices": mapping.response_indices.astype(int).tolist(),
                    "vertexIndices": mapping.vertex_indices.astype(int).tolist(),
                    "weights": mapping.vertex_normalized_weights.astype(float).tolist(),
                }
                for hemi, mapping in projection.mappings.items()
            },
        }
    return {
        "schemaVersion": 1,
        "subjectId": "s1",
        "responseSpace": RESPONSE_SPACE,
        "fLocThreshold": THRESHOLD,
        "fLocThresholdOperator": ">",
        "responseOrdering": "zyx-c-order",
        "mapper": "linenn",
        "volumeShapeXyz": list(VOLUME_SHAPE_XYZ),
        "normalizationPolicy": "per-surface-vertex",
        "normalization": (
            "For every surface vertex represented in an ROI mapping, the weights "
            "arriving from selected response voxels sum to 1."
        ),
        "rois": rois,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    path.write_text(payload + "\n", encoding="utf-8")


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.asanyarray(arrays[name])
            if array.dtype.hasobject:
                raise BrainArtifactValidationError(f"Object dtype is not portable: {name}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)


def write_full_derived_mapping(path: Path, projections: Mapping[str, RoiProjection]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for roi, projection in projections.items():
        arrays[f"{roi}_coordinates_zyx"] = projection.selection.coordinates_zyx
        arrays[f"{roi}_linear_indices_xyz"] = projection.selection.linear_indices_xyz
        arrays[f"{roi}_unmapped_response_indices"] = np.asarray(
            projection.unmapped_response_indices, dtype="<i4"
        )
        for hemi, mapping in projection.mappings.items():
            arrays[f"{roi}_{hemi}_response_indices"] = mapping.response_indices
            arrays[f"{roi}_{hemi}_vertex_indices"] = mapping.vertex_indices
            arrays[f"{roi}_{hemi}_raw_weights"] = mapping.raw_weights
            arrays[f"{roi}_{hemi}_response_normalized_weights"] = (
                mapping.response_normalized_weights
            )
            arrays[f"{roi}_{hemi}_vertex_normalized_weights"] = (
                mapping.vertex_normalized_weights
            )
    write_deterministic_npz(path, arrays)


def _source_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "fileName": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    selections, source_paths = load_roi_selections(args.nsd_data_root, args.body_labels)
    tensor_paths = validate_training_tensors(args.tensor_root, selections)
    source_paths.update(tensor_paths)
    projections, vertex_counts = load_projections(selections, args.mapper)
    source_paths["linenn_mapper"] = args.mapper

    expected_unmapped = {"ffa": (), "eba": (42, 66, 120), "ppa": ()}
    for roi, expected in expected_unmapped.items():
        actual = projections[roi].unmapped_response_indices
        if actual != expected:
            raise BrainArtifactValidationError(
                f"{roi.upper()} unmapped responses are {list(actual)}; audited contract is "
                f"{list(expected)}"
            )

    args.public_out.mkdir(parents=True, exist_ok=True)
    args.derived_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nsd-s1-surfaces-") as temporary:
        surfaces, surface_member_hashes = extract_surface_files(
            args.mentor_archive, Path(temporary)
        )
        surface_metadata: dict[str, Any] = {}
        for variant in ("inflated", "pial"):
            glb_path = args.public_out / f"{variant}.glb"
            hemispheres = export_surface_glb(
                surfaces[variant], glb_path, vertex_counts
            )
            surface_metadata[variant] = {
                "url": f"{variant}.glb",
                "sha256": sha256_file(glb_path),
                "bytes": glb_path.stat().st_size,
                "normalsIncluded": False,
                "hemispheres": {
                    hemi: {"node": hemi, **values}
                    for hemi, values in hemispheres.items()
                },
            }

    projection_path = args.public_out / "projection.json"
    write_json(projection_path, projection_document(projections))
    derived_mapping_path = args.derived_out / "subject1_floc_t7_linenn_full.npz"
    write_full_derived_mapping(derived_mapping_path, projections)

    source_paths["mentor_archive"] = args.mentor_archive
    sources = _source_records(source_paths)
    for name, digest in sorted(surface_member_hashes.items()):
        variant, hemi = name.split("_", maxsplit=1)
        sources[f"mentor_archive_member_{name}"] = {
            "fileName": f"{hemi}.{variant}",
            "sha256": digest,
        }

    roi_manifest: dict[str, Any] = {}
    for roi, spec in ROI_SPECS.items():
        projection = projections[roi]
        mapped_count = spec["response_count"] - len(projection.unmapped_response_indices)
        roi_manifest[roi] = {
            "label": spec["label"],
            "color": spec["color"],
            "domain": spec["domain"],
            "labelValues": list(spec["label_values"]),
            "thresholdOperator": ">",
            "threshold": THRESHOLD,
            "responseCount": spec["response_count"],
            "mappedResponseCount": mapped_count,
            "unmappedResponseIndices": list(projection.unmapped_response_indices),
            "orderedCoordinateSha256": EXPECTED_COORDINATE_SHA256[roi],
        }

    manifest = {
        "schemaVersion": 1,
        "subjectId": "s1",
        "responseSpace": RESPONSE_SPACE,
        "fLocThreshold": THRESHOLD,
        "fLocThresholdOperator": ">",
        "responseOrdering": "zyx-c-order",
        "mapper": "linenn",
        "volumeShapeXyz": list(VOLUME_SHAPE_XYZ),
        "normalizationPolicy": "per-surface-vertex",
        "subject": {
            "id": "nsd-subj01",
            "label": "NSD Subject 1",
            "responseSpace": RESPONSE_SPACE,
        },
        "description": (
            "Predicted, not measured, responses for NSD Subject 1; only FFA, EBA, "
            "and PPA model targets are represented."
        ),
        "volumeSpace": {
            "name": "NSD func1pt8",
            "shapeXyz": list(VOLUME_SHAPE_XYZ),
            "sourceShapeZyx": list(VOLUME_SHAPE_ZYX),
            "sourceAxisOrder": "zyx",
            "responseOrdering": (
                "C-order over the source (z,y,x) volume after cortical-mask expansion"
            ),
        },
        "meshSpace": {
            "source": "FreeSurfer Subject 1 native RAS",
            "gltfTransform": "(x,y,z) -> (x,z,-y)",
            "mapper": "pycortex linenn",
            "vertexOrderingPreserved": True,
            "recentered": False,
        },
        "projectionUrl": "projection.json",
        "projectionSha256": sha256_file(projection_path),
        "projectionBytes": projection_path.stat().st_size,
        "surfaces": surface_metadata,
        "rois": roi_manifest,
        "provenance": {
            "selection": (
                "Training-era Subject 1 cortical-index labels and t-values, fLoc ROI "
                "labels, strict t > 7."
            ),
            "excludedSource": (
                "surfacevoxels_layerB2.mgz is an NSD sampling test pattern and is not "
                "used as an index map."
            ),
            "sources": sources,
            "derivedFullMapping": {
                "fileName": derived_mapping_path.name,
                "sha256": sha256_file(derived_mapping_path),
                "bytes": derived_mapping_path.stat().st_size,
                "public": False,
            },
        },
    }
    manifest_path = args.public_out / "manifest.json"
    write_json(manifest_path, manifest)

    report = {
        "manifest": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "projection": str(projection_path),
        "projectionSha256": sha256_file(projection_path),
        "derivedMapping": str(derived_mapping_path),
        "rois": {
            roi: {
                "responses": len(projection.selection.coordinates_zyx),
                "mapped": len(projection.selection.coordinates_zyx)
                - len(projection.unmapped_response_indices),
                "unmappedResponseIndices": list(projection.unmapped_response_indices),
            }
            for roi, projection in projections.items()
        },
    }
    write_json(args.derived_out / "build-report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mentor-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--nsd-data-root", type=Path, default=DEFAULT_NSD_DATA_ROOT)
    parser.add_argument("--body-labels", type=Path, default=DEFAULT_BODY_LABELS)
    parser.add_argument("--mapper", type=Path, default=DEFAULT_MAPPER)
    parser.add_argument("--tensor-root", type=Path, default=DEFAULT_TENSOR_ROOT)
    parser.add_argument("--public-out", type=Path, default=DEFAULT_PUBLIC_OUT)
    parser.add_argument("--derived-out", type=Path, default=DEFAULT_DERIVED_OUT)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
