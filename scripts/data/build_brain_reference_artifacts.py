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
import re
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROIS = ("ffa", "ppa", "eba")
COUNTS = {"ffa": 156, "ppa": 170, "eba": 518}
FULL_COUNTS = {"ffa": 716, "ppa": 753, "eba": 2045}
SUBJECT_COUNTS = {
    "s1": {"ffa": 156, "ppa": 170, "eba": 518},
    "s2": {"ffa": 218, "ppa": 224, "eba": 526},
    "s5": {"ffa": 271, "ppa": 295, "eba": 682},
    "s7": {"ffa": 71, "ppa": 64, "eba": 319},
}
INFERENCE_PREPROCESSING = {
    "image_mode": "RGB", "square_padding": True, "pad_color": 0,
    "padding_placement": "center_floor_offset", "backbone_transforms": "unmodified_cortex_model_transforms",
    "inference_call": "ReadoutModel.inference(return_logits=True, do_square_padding=True)",
    "voxel_normalization": "none", "tail_policy": "none",
}
INFERENCE_NUMERIC_POLICY = {
    "policy": "fp32_no_tf32_v1", "output_dtype": "float32",
    "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
    "cuda_autocast_enabled": False, "cpu_autocast_enabled": False,
}
OFFSETS = {"ffa": 0, "ppa": 156, "eba": 326}
REFERENCE_COUNT = 185
ROW_WIDTH = 844
RESPONSE_SPACE = "nsd_1000_subject1_floc_t7_v1"
EXPECTED_REFERENCE_VERSION = "ed023e59bd270e210bb8350d348eaaf223d3992214adeb2163a496cc808cefa7"
BASELINE_BRAIN_MANIFEST_SHA256 = "86ac452a2c5badeb436a219be70e10eef47825b749bcf8abdf67b1b9d78b301d"
MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*(?:\.[0-9]+)?$")
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


def canonical_json_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_model_id(model: str) -> None:
    # A decimal is part of the exact registered TopoNets tau=0.5 identifier.
    # Separators, traversal, leading/trailing dots, and arbitrary suffixes fail.
    if not isinstance(model, str) or MODEL_ID_RE.fullmatch(model) is None:
        raise BrainReferenceValidationError("Unsafe model identifier.")


def expected_prediction_policy(model: str) -> dict:
    # The deployed VOne-RN50 uses resnet50_ns: its runtime noise modes are empty.
    # Only these two deployed constructors retain active neuronal sampling.
    stochastic = model in {"vone_alexnet", "vone_cornet_s"}
    return {
        "response_policy": "seeded_stochastic_sampling" if stochastic else "evaluation_mode",
        "deterministic_per_stimulus": not stochastic,
        "noise_modes": ["neuronal"] if stochastic else [],
    }


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
    if not rows or not rows[0] or rows[0][0] != "image" or len(rows[0]) != len(set(rows[0])):
        raise BrainReferenceValidationError(f"{roi}: source header must have unique columns beginning with 'image'.")
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
        validate_model_id(model)
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


def validate_spatial_expansion(
    *,
    artifact_root: Path,
    raw_root: Path,
    baseline_artifact_root: Path,
    audit_path: Path,
    image_root: Path = REPO_ROOT / "public/images",
) -> tuple[dict, dict, dict[str, dict[str, np.ndarray]]]:
    """Independently audit new prediction triads before any spatial export.

    The existing audited identities remain the trust anchor. Each extension
    must bind canonical stimulus bytes, full pooled voxel-column order, genuine
    checkpoint bytes, explicit FP32 numeric policy, and the unmodified raw
    matrix to its new reference entry. Only additions receive this new precision
    contract; no precision claim is retroactively made about the baseline.
    An arbitrary added model name or same-width tensor is not sufficient.
    """
    audit = load_audit_constants(audit_path)
    baseline = json.loads((baseline_artifact_root / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    baseline_models = set(audit["AUDITED_CHECKPOINT_SHA256"])
    models = reference.get("supported_models")
    version = reference.get("reference_version")
    if (
        baseline.get("reference_version") != EXPECTED_REFERENCE_VERSION
        or set(baseline.get("supported_models", [])) != baseline_models
        or reference.get("dataset") != "nsd_1000"
        or reference.get("image_count") != REFERENCE_COUNT
        or reference.get("reference_count") != REFERENCE_COUNT
        or reference.get("voxel_counts") != FULL_COUNTS
        or reference.get("image_ids_sha256") != canonical_json_sha256(canonical_image_ids())
        or reference.get("checksum") != version
        or not isinstance(version, str)
        or re.fullmatch(r"[a-f0-9]{64}", version) is None
        or version == EXPECTED_REFERENCE_VERSION
        or not isinstance(models, list)
        or len(models) != len(set(models))
        or not baseline_models < set(models)
    ):
        raise BrainReferenceValidationError("Spatial expansion must extend the exact pinned baseline reference identity.")
    for model in models:
        validate_model_id(model)
    if set(reference.get("artifacts", {})) != set(models):
        raise BrainReferenceValidationError("Spatial expansion artifact/model coverage is incomplete.")
    for model in sorted(baseline_models):
        if reference["artifacts"][model] != baseline["artifacts"][model]:
            raise BrainReferenceValidationError(f"{model}: expansion changed a baseline reference artifact record.")
        for roi in ROIS:
            key = f"{model}/{roi}"
            if reference["source_selection"].get(key) != baseline["source_selection"].get(key):
                raise BrainReferenceValidationError(f"{key}: expansion changed baseline source selection.")
            old_entry = baseline["artifacts"][model][roi]
            old_path = (artifact_root / old_entry["path"]).resolve()
            if not old_path.is_relative_to(artifact_root.resolve()) or sha256(old_path.read_bytes()) != old_entry["sha256"]:
                raise BrainReferenceValidationError(f"{key}: expansion changed a baseline artifact payload.")

    extension = reference.get("coverage_extension", {})
    if (
        extension.get("policy") != "append_only_preserve_baseline_artifacts_v1"
        or extension.get("parent_reference_version") != EXPECTED_REFERENCE_VERSION
        or set(extension.get("baseline_models", [])) != baseline_models
        or set(extension.get("added_models", [])) != set(models) - baseline_models
        or set(extension.get("new_inference_provenance_sha256", {})) != set(models) - baseline_models
        or not isinstance(extension.get("prediction_policies"), dict)
        or set(extension["prediction_policies"]) != set(models) - baseline_models
    ):
        raise BrainReferenceValidationError("Spatial expansion lacks exact append-only baseline lineage.")

    expected_images = [{"id": image, "sha256": sha256((image_root / image).read_bytes())} for image in canonical_image_ids()]
    layer_path = audit_path.parents[1] / "constants/best_readout_layers.py"
    layer_tree = ast.parse(layer_path.read_text(encoding="utf-8"))
    layers = next((ast.literal_eval(node.value) for node in layer_tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "best_readout_layers" for target in node.targets)), None)
    if not isinstance(layers, dict):
        raise BrainReferenceValidationError("Could not read canonical NSD readout-layer metadata.")
    precision_path = audit_path.parents[1] / "utils/inference_precision.py"
    if not precision_path.is_file():
        raise BrainReferenceValidationError("The verified FP32 inference precision helper is unavailable.")
    precision_hash = sha256(precision_path.read_bytes())
    checkpoint_hashes = {model: list(values) for model, values in audit["AUDITED_CHECKPOINT_SHA256"].items()}
    provenance_hashes = {}
    prediction_policies = {}
    matrices = {}
    for model in sorted(set(models) - baseline_models):
        model_root = (raw_root / model).resolve()
        if not model_root.is_relative_to(raw_root.resolve()):
            raise BrainReferenceValidationError("New model source path escapes the raw root.")
        proof_path = model_root / "provenance.json"
        proof_bytes = proof_path.read_bytes()
        proof = json.loads(proof_bytes)
        policy = expected_prediction_policy(model)
        numeric_policy = proof.get("numeric_policy")
        source_hashes = proof.get("core_inference_source_sha256")
        if (not isinstance(numeric_policy, dict)
                or numeric_policy != INFERENCE_NUMERIC_POLICY
                or any(type(numeric_policy.get(key)) is not type(value) for key, value in INFERENCE_NUMERIC_POLICY.items())
                or proof.get("precision_helper_sha256") != precision_hash
                or not isinstance(source_hashes, dict)
                or source_hashes.get("cortex/utils/inference_precision.py") != precision_hash
                or extension["prediction_policies"][model] != {
                    "numeric_policy": INFERENCE_NUMERIC_POLICY, "precision_helper_sha256": precision_hash,
                }):
            raise BrainReferenceValidationError(f"{model}: new spatial predictions lack the exact FP32 policy/helper identity.")
        if (
            proof.get("schema_version") != 1
            or proof.get("model") != model
            or proof.get("dataset") != "nsd_1000"
            or proof.get("reference_count") != REFERENCE_COUNT
            or proof.get("image_ids") != canonical_image_ids()
            or proof.get("images") != expected_images
            or proof.get("image_set_sha256") != canonical_json_sha256(expected_images)
            or proof.get("subject_order") != ["s1", "s2", "s5", "s7"]
            or proof.get("subject_voxel_counts") != SUBJECT_COUNTS
            or proof.get("voxel_counts") != FULL_COUNTS
            or proof.get("preprocessing") != INFERENCE_PREPROCESSING
            or proof.get("seed") != 0
            or not isinstance(proof.get("backbone_state_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", proof["backbone_state_sha256"]) is None
            or sha256(proof_bytes) != extension["new_inference_provenance_sha256"][model]
            or any(proof.get(key) != value for key, value in policy.items())
            or ("model_prediction_policies" in extension and extension["model_prediction_policies"].get(model) != policy)
            or set(proof.get("rois", {})) != set(ROIS)
        ):
            raise BrainReferenceValidationError(f"{model}: inference provenance does not establish the canonical NSD/Murty identity.")
        checkpoint_hashes[model] = []
        matrices[model] = {}
        for roi in ROIS:
            record = proof["rois"][roi]
            entry = reference["artifacts"][model][roi]
            layer = layers.get("nsd_1000", {}).get(roi, {}).get(model)
            checkpoint_path = Path(record.get("checkpoint_path", ""))
            expected_filename = f"{model}_nsd_1000_all_{roi}_{layer}.model"
            columns = record.get("voxel_columns")
            if (
                layer is None
                or record.get("readout_layer") != layer
                or not checkpoint_path.is_absolute()
                or checkpoint_path.name != expected_filename
                or not checkpoint_path.is_file()
                or sha256(checkpoint_path.read_bytes()) != record.get("checkpoint_sha256")
                or record.get("path") != entry.get("source_file")
                or record.get("sha256") != entry.get("source_sha256")
                or record.get("output_shape") != [REFERENCE_COUNT, FULL_COUNTS[roi]]
                or not isinstance(columns, list)
                or len(columns) != FULL_COUNTS[roi]
                or canonical_json_sha256(columns) != audit["AUDITED_VOXEL_ORDER_SHA256"][roi]
                or record.get("voxel_order_sha256") != audit["AUDITED_VOXEL_ORDER_SHA256"][roi]
            ):
                raise BrainReferenceValidationError(f"{model}/{roi}: checkpoint lineage or full voxel order is not audited.")
            if reference["source_selection"][f"{model}/{roi}"].get("selected") != entry["source_file"]:
                raise BrainReferenceValidationError(f"{model}/{roi}: selected source does not match the artifact.")
            matrices[model][roi] = validated_roi_matrix(artifact_root, model_root, entry, roi, audit["AUDITED_VOXEL_ORDER_SHA256"][roi])
            checkpoint_hashes[model].append(record["checkpoint_sha256"])
        provenance_hashes[model] = sha256(proof_bytes)
        prediction_policies[model] = policy
    contract = {
        "schema_version": 1,
        "baseline_reference_version": EXPECTED_REFERENCE_VERSION,
        "reference_version": version,
        "dataset": "nsd_1000",
        "subject_id": "s1",
        "response_space": RESPONSE_SPACE,
        "subject_column_policy": "audited_s1_prefix_from_pooled_nsd_readout",
        "image_ids_sha256": canonical_json_sha256(canonical_image_ids()),
        "full_voxel_counts": FULL_COUNTS,
        "subject1_voxel_counts": COUNTS,
        "voxel_order_sha256": audit["AUDITED_VOXEL_ORDER_SHA256"],
        "checkpoint_sha256_by_model": checkpoint_hashes,
        "inference_provenance_sha256_by_model": provenance_hashes,
        "model_prediction_policies": prediction_policies,
        "new_inference_numeric_policy": INFERENCE_NUMERIC_POLICY,
        "precision_helper_sha256": precision_hash,
    }
    return reference, contract, matrices


def append_reference_brain_assets(
    *,
    artifact_root: Path,
    raw_root: Path,
    baseline_artifact_root: Path,
    audit_path: Path,
    baseline_brain_root: Path,
    output_root: Path,
    image_root: Path = REPO_ROOT / "public/images",
) -> dict:
    """Stage a validated extension; retain all 26 baseline payload bytes.

    This function does not promote assets, modify Core pins, or replace an
    existing directory. Its spatial-audit.json needs independent review and a
    deliberate Core digest pin before live uploads can gain anatomical support.
    """
    if output_root.exists():
        raise BrainReferenceValidationError("Expansion output must be a new staging directory.")
    baseline_bytes = (baseline_brain_root / "manifest.json").read_bytes()
    if sha256(baseline_bytes) != BASELINE_BRAIN_MANIFEST_SHA256:
        raise BrainReferenceValidationError("Baseline brain manifest does not match the reviewed digest.")
    baseline = json.loads(baseline_bytes)
    reference, contract, matrices = validate_spatial_expansion(
        artifact_root=artifact_root, raw_root=raw_root, baseline_artifact_root=baseline_artifact_root,
        audit_path=audit_path, image_root=image_root,
    )
    if set(baseline["models"]) != set(contract["checkpoint_sha256_by_model"]) - set(matrices):
        raise BrainReferenceValidationError("Baseline brain model coverage differs from the pinned audit.")
    payloads = {}
    for model, entry in baseline["models"].items():
        validate_model_id(model)
        if entry["url"] != f"{model}.f32":
            raise BrainReferenceValidationError("Unsafe baseline brain child path.")
        payload = (baseline_brain_root / entry["url"]).read_bytes()
        if len(payload) != REFERENCE_COUNT * ROW_WIDTH * 4 or sha256(payload) != entry["sha256"]:
            raise BrainReferenceValidationError(f"{model}: baseline brain payload changed.")
        payloads[model] = payload
    entries = dict(baseline["models"])
    for model, roi_matrices in matrices.items():
        payload = np.ascontiguousarray(np.concatenate([roi_matrices[roi] for roi in ROIS], axis=1), dtype="<f4").tobytes(order="C")
        payloads[model] = payload
        entries[model] = {
            "url": f"{model}.f32", "bytes": len(payload), "sha256": sha256(payload),
            "provenance": {roi: {
                "artifactSha256": reference["artifacts"][model][roi]["sha256"],
                "sourceSha256": reference["artifacts"][model][roi]["source_sha256"],
                "voxelOrderSha256": contract["voxel_order_sha256"][roi],
                "checkpointSha256": contract["checkpoint_sha256_by_model"][model][ROIS.index(roi)],
            } for roi in ROIS},
        }
    index = {**baseline, "referenceVersion": reference["reference_version"], "models": entries}
    output_root.mkdir(parents=True)
    try:
        for model, payload in payloads.items():
            (output_root / f"{model}.f32").write_bytes(payload)
        (output_root / "manifest.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        (output_root / "spatial-audit.json").write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return index


def validate_spatial_standardization(
    *, artifact_root: Path, raw_root: Path, baseline_root: Path, audit_path: Path,
    image_root: Path = REPO_ROOT / "public/images",
) -> tuple[dict, dict, dict[str, dict[str, np.ndarray]]]:
    try:
        from .spatial_standardization import validate_spatial_standardization as validate
    except ImportError:  # pragma: no cover - direct script execution
        from spatial_standardization import validate_spatial_standardization as validate
    return validate(artifact_root=artifact_root, raw_root=raw_root, baseline_root=baseline_root,
                    audit_path=audit_path, image_root=image_root)


def standardize_reference_brain_assets(
    *, artifact_root: Path, raw_root: Path, baseline_root: Path, audit_path: Path,
    output_root: Path, image_root: Path = REPO_ROOT / "public/images",
) -> dict:
    """Freshly export all 126 audited matrices into a new, unpromoted directory."""
    try:
        from .spatial_standardization import PARENT_BRAIN_MANIFEST_SHA256, checked_document, validate_spatial_standardization
    except ImportError:  # pragma: no cover - direct script execution
        from spatial_standardization import PARENT_BRAIN_MANIFEST_SHA256, checked_document, validate_spatial_standardization
    if output_root.exists():
        raise BrainReferenceValidationError("Standardization output must be a new staging directory.")
    reference, contract, matrices = validate_spatial_standardization(
        artifact_root=artifact_root, raw_root=raw_root, baseline_root=baseline_root,
        audit_path=audit_path, image_root=image_root,
    )
    parent_root = baseline_root / "public/data/brain/reference"
    parent = checked_document(parent_root / "manifest.json", PARENT_BRAIN_MANIFEST_SHA256)
    entries = {}
    # Preserve unrelated parent files, but every known model payload below is
    # freshly derived from the newly validated raw matrix, never copied as data.
    shutil.copytree(parent_root, output_root)
    try:
        for model in sorted(matrices):
            payload = np.ascontiguousarray(np.concatenate([matrices[model][roi] for roi in ROIS], axis=1), dtype="<f4").tobytes(order="C")
            (output_root / f"{model}.f32").write_bytes(payload)
            entries[model] = {
                "url": f"{model}.f32", "bytes": len(payload), "sha256": sha256(payload),
                "provenance": {roi: {
                    "artifactSha256": reference["artifacts"][model][roi]["sha256"],
                    "sourceSha256": reference["artifacts"][model][roi]["source_sha256"],
                    "voxelOrderSha256": contract["voxel_order_sha256"][roi],
                    "checkpointSha256": contract["checkpoint_sha256_by_model"][model][ROIS.index(roi)],
                } for roi in ROIS},
            }
        index = {**parent, "referenceVersion": reference["reference_version"], "models": entries}
        (output_root / "manifest.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        (output_root / "spatial-audit.json").write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts/reference")
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data/image_by_model")
    parser.add_argument("--audit-path", type=Path, default=REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "public/data/brain/reference")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--append-baseline-brain", type=Path, help="Explicitly stage an audited extension of this pinned baseline brain bundle.")
    mode.add_argument("--standardize-from-baseline", type=Path, help="Freshly stage every model using full-roster FP32 proofs and this reviewed rollback snapshot.")
    parser.add_argument("--baseline-artifact-root", type=Path, default=REPO_ROOT / "artifacts/reference")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "public/images")
    args = parser.parse_args()
    if args.standardize_from_baseline:
        manifest = standardize_reference_brain_assets(
            artifact_root=args.artifact_root, raw_root=args.raw_root, audit_path=args.audit_path,
            baseline_root=args.standardize_from_baseline, output_root=args.output_root, image_root=args.image_root,
        )
    elif args.append_baseline_brain:
        manifest = append_reference_brain_assets(
            artifact_root=args.artifact_root, raw_root=args.raw_root, audit_path=args.audit_path,
            baseline_artifact_root=args.baseline_artifact_root, baseline_brain_root=args.append_baseline_brain,
            output_root=args.output_root, image_root=args.image_root,
        )
    else:
        manifest = build_reference_brain_assets(args.artifact_root, args.raw_root, args.audit_path, args.output_root)
    print(f"Validated {len(manifest['models'])} models × 185 reference images; exported raw Subject 1 predictions.")
    print(f"Manifest SHA-256: {sha256((args.output_root / 'manifest.json').read_bytes())}")


if __name__ == "__main__":
    main()
