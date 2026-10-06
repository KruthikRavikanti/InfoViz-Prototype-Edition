#!/usr/bin/env python3
"""Export audited Murty-185 predictions for the NSD Subject 1 brain map.

This is a lossless subset of the existing raw float32 reference tensors, not
new inference, a cluster-space reconstruction, or measured fMRI. The original
immutable comparison bundle and anatomy artifacts are not modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROIS = ("ffa", "ppa", "eba")
COUNTS = {"ffa": 156, "ppa": 170, "eba": 518}
FULL_COUNTS = {"ffa": 716, "ppa": 753, "eba": 2045}
OFFSETS = {"ffa": 0, "ppa": 156, "eba": 326}
REFERENCE_COUNT = 185
ROW_WIDTH = 844
RESPONSE_SPACE = "nsd_1000_subject1_floc_t7_v1"
EXPECTED_REFERENCE_VERSION = "ed023e59bd270e210bb8350d348eaaf223d3992214adeb2163a496cc808cefa7"
COORDINATE_HASHES = {
    "ffa": "ce13c8bcd14cc54865b90cc4d9546559afd92269ad2bdc9a66df9aa327ef8a88",
    "ppa": "e0b164fe5a54d564b09e4dc25b4d456bb966933870e3dbb9aebd93d73657282e",
    "eba": "440e3d36e725f44df7a4bfc6aeab9cc3fc180853adde80843439393698c36fbe",
}


class BrainReferenceValidationError(ValueError):
    pass


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_image_ids() -> list[str]:
    return [f"image_{index:03d}.png" for index in range(1, REFERENCE_COUNT + 1)]


def load_audit_constants(path: Path) -> dict:
    """Read only literal audit constants, without importing the inference stack."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"AUDITED_REFERENCE_VERSION", "AUDITED_VOXEL_ORDER_SHA256", "AUDITED_CHECKPOINT_SHA256"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                result[name] = ast.literal_eval(node.value)
    if set(result) != wanted or result["AUDITED_REFERENCE_VERSION"] != EXPECTED_REFERENCE_VERSION:
        raise BrainReferenceValidationError("Core audit does not cover the pinned reference bundle.")
    return result


def validated_roi_matrix(artifact_root: Path, raw_root: Path, entry: dict, roi: str, voxel_order_hash: str) -> np.ndarray:
    artifact_path = (artifact_root / entry["path"]).resolve()
    if not artifact_path.is_relative_to(artifact_root.resolve()):
        raise BrainReferenceValidationError("Artifact path escapes the reference bundle.")
    artifact_bytes = artifact_path.read_bytes()
    if sha256(artifact_bytes) != entry["sha256"]:
        raise BrainReferenceValidationError(f"{roi}: reference artifact checksum mismatch.")
    source_path = raw_root / entry["source_file"]
    if source_path.name != entry["source_file"] or sha256(source_path.read_bytes()) != entry["source_sha256"]:
        raise BrainReferenceValidationError(f"{roi}: raw source checksum mismatch.")
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    columns = rows[0][1:]
    order_hash = sha256(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    if order_hash != voxel_order_hash:
        raise BrainReferenceValidationError(f"{roi}: full voxel order is not the audited NSD order.")
    expected_prefix = [f"s1_{index}" for index in range(COUNTS[roi])]
    if columns[:COUNTS[roi]] != expected_prefix or [c for c in columns if c.startswith("s1_")] != expected_prefix:
        raise BrainReferenceValidationError(f"{roi}: Subject 1 is not the exact leading ordered prefix.")
    if [row[0] for row in rows[1:]] != canonical_image_ids():
        raise BrainReferenceValidationError(f"{roi}: source reference IDs/order changed.")
    raw_matrix = np.asarray([row[1:] for row in rows[1:]], dtype=np.float32)
    with np.load(artifact_path, allow_pickle=False) as arrays:
        matrix = arrays["reference_voxels"]
        if arrays["image_ids"].tolist() != canonical_image_ids():
            raise BrainReferenceValidationError(f"{roi}: artifact reference IDs/order changed.")
        if matrix.dtype != np.float32 or matrix.shape != (REFERENCE_COUNT, FULL_COUNTS[roi]) or not np.isfinite(matrix).all():
            raise BrainReferenceValidationError(f"{roi}: raw reference matrix shape/type/values are invalid.")
        if not np.array_equal(matrix, raw_matrix):
            raise BrainReferenceValidationError(f"{roi}: artifact is not the raw prediction matrix.")
        return np.ascontiguousarray(matrix[:, :COUNTS[roi]], dtype="<f4")


def build_reference_brain_assets(artifact_root: Path, raw_root: Path, audit_path: Path, output_root: Path) -> dict:
    audit = load_audit_constants(audit_path)
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("reference_version") != EXPECTED_REFERENCE_VERSION or manifest.get("dataset") != "nsd_1000" or manifest.get("reference_count") != REFERENCE_COUNT:
        raise BrainReferenceValidationError("Reference bundle identity/dataset/count is incompatible.")
    models = manifest["supported_models"]
    if set(models) != set(audit["AUDITED_CHECKPOINT_SHA256"]) or len(models) != len(set(models)):
        raise BrainReferenceValidationError("All and only audited models must be exported.")
    # Validate every model before publishing anything: a bad source cannot leave
    # a partially validated bundle or cause a guessed column interpretation.
    payloads = {}
    entries = {}
    for model in sorted(models):
        if not model.replace("_", "").isalnum():
            raise BrainReferenceValidationError("Unsafe model identifier.")
        matrices = []
        provenance = {}
        for roi in ROIS:
            entry = manifest["artifacts"][model][roi]
            matrices.append(validated_roi_matrix(artifact_root, raw_root, entry, roi, audit["AUDITED_VOXEL_ORDER_SHA256"][roi]))
            provenance[roi] = {
                "artifactSha256": entry["sha256"],
                "sourceSha256": entry["source_sha256"],
                "voxelOrderSha256": audit["AUDITED_VOXEL_ORDER_SHA256"][roi],
                "checkpointSha256": audit["AUDITED_CHECKPOINT_SHA256"][model][ROIS.index(roi)],
            }
        payload = np.ascontiguousarray(np.concatenate(matrices, axis=1), dtype="<f4").tobytes(order="C")
        payloads[model] = payload
        entries[model] = {"url": f"{model}.f32", "bytes": len(payload), "sha256": sha256(payload), "provenance": provenance}
    index = {
        "schemaVersion": 1,
        "dataset": "nsd_1000",
        "stimulusSet": "Murty-185",
        "responseKind": "model_prediction",
        "subjectId": "s1",
        "responseSpace": RESPONSE_SPACE,
        "referenceVersion": EXPECTED_REFERENCE_VERSION,
        "normalization": "none",
        "responseOrdering": "zyx-c-order",
        "subjectColumnPolicy": "audited_s1_prefix_from_pooled_nsd_readout",
        "dtype": "float32-le",
        "rowWidth": ROW_WIDTH,
        "referenceCount": REFERENCE_COUNT,
        "imageIds": canonical_image_ids(),
        "roiLayout": {roi: {"offset": OFFSETS[roi], "count": COUNTS[roi], "orderedCoordinateSha256": COORDINATE_HASHES[roi]} for roi in ROIS},
        "models": entries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for model, payload in payloads.items():
        (output_root / f"{model}.f32").write_bytes(payload)
    (output_root / "manifest.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts/reference")
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data/image_by_model")
    parser.add_argument("--audit-path", type=Path, default=REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "public/data/brain/reference")
    args = parser.parse_args()
    manifest = build_reference_brain_assets(args.artifact_root, args.raw_root, args.audit_path, args.output_root)
    print(f"Validated {len(manifest['models'])} models × 185 reference images; exported raw Subject 1 predictions.")
    print(f"Manifest SHA-256: {sha256((args.output_root / 'manifest.json').read_bytes())}")


if __name__ == "__main__":
    main()
