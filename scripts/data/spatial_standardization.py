"""Independent full-roster FP32 spatial audit; historical append rules stay separate."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np

try:
    from . import build_brain_reference_artifacts as brain
except ImportError:  # Direct builder CLI.
    import build_brain_reference_artifacts as brain


PARENT_REFERENCE_VERSION = "a862bba8496384d80de1b1990152b2b683e4325a52b0880d85783072172246d0"
PARENT_MANIFEST_SHA256 = "9258d85bfb911db0a4dc959659d0e8b57e69957fc73f671416fdeea9a96c7475"
PARENT_BRAIN_MANIFEST_SHA256 = "ebe0a4da8cd42eea2abd94060724af6efca48520efacdcf5e48f783792617c81"
PARENT_ATLAS_MANIFEST_SHA256 = "ba589e7108173009c33bc1489a04423aac69b2d2e3ed2a6ad9c4ace1a2100418"
PARENT_SPATIAL_AUDIT_SHA256 = "549740cef1493cb8a811444c4e3f777a193286ffd7affd7595d09035858b70d9"
POLICY = "full_roster_fp32_regeneration_v1"
ANALYSIS_INPUT_POLICY = "canonical_float32_promoted_float64_v1"
MODEL_COUNT = 126
ANATOMY_SHA256 = {
    "public/data/brain/nsd-subj01/manifest.json": "3ac10989e3ac26e8388d83b47d200a80d99a0b4532ff90386788b3285700f7ce",
    "public/data/brain/nsd-subj01/projection.json": "4c0bf0d19aa6ceda5ae61f352f26b874f32450f49d0945b7c1022aaaa703618f",
    "public/data/brain/nsd-subj01/inflated.glb": "62d9077896094e5ea196058379c2f2e85d6f5749207aa677c7355b5959a18241",
    "public/data/brain/nsd-subj01/pial.glb": "8713d5d2f8f8ec24b42a65d9bcf9616561a942624610357d85224495b9c4f577",
}


def checked_document(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    if brain.sha256(payload) != expected_sha256:
        raise brain.BrainReferenceValidationError(f"Reviewed parent payload changed: {path.name}.")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise brain.BrainReferenceValidationError("Reviewed parent document must be an object.")
    return value


def exact_map(value: object, models: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != models:
        raise brain.BrainReferenceValidationError(f"Standardization needs exact full-roster {label}.")
    return value


def validate_spatial_standardization(
    *,
    artifact_root: Path,
    raw_root: Path,
    baseline_root: Path,
    audit_path: Path,
    image_root: Path = brain.REPO_ROOT / "public/images",
) -> tuple[dict, dict, dict[str, dict[str, np.ndarray]]]:
    """Bind every fresh prediction to the reviewed parent's exact voxel/readout identity.

    This is regeneration, not append-only preservation. All 126 models need
    fresh canonical-image/raw-column/checkpoint and precision-helper proofs.
    The original 26 checkpoint and pooled-column audit constants are untouched.
    """
    audit = brain.load_audit_constants(audit_path)
    parent = checked_document(baseline_root / "artifacts/reference/manifest.json", PARENT_MANIFEST_SHA256)
    parent_brain = checked_document(baseline_root / "public/data/brain/reference/manifest.json", PARENT_BRAIN_MANIFEST_SHA256)
    parent_atlas = checked_document(baseline_root / "public/data/voxel_selectivity_atlas/manifest.json", PARENT_ATLAS_MANIFEST_SHA256)
    parent_audit = checked_document(baseline_root / "public/data/brain/reference/spatial-audit.json", PARENT_SPATIAL_AUDIT_SHA256)
    snapshot_bytes = (baseline_root / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_bytes)
    models = parent.get("supported_models")
    if (not isinstance(models, list) or len(models) != MODEL_COUNT or len(set(models)) != MODEL_COUNT
            or parent.get("reference_version") != PARENT_REFERENCE_VERSION
            or parent_brain.get("referenceVersion") != PARENT_REFERENCE_VERSION
            or parent_atlas.get("referenceVersion") != PARENT_REFERENCE_VERSION
            or parent_audit.get("reference_version") != PARENT_REFERENCE_VERSION
            or snapshot.get("reference_version") != PARENT_REFERENCE_VERSION
            or set(snapshot.get("models", [])) != set(models)):
        raise brain.BrainReferenceValidationError("Standardization parent must be the exact reviewed 126-model release.")
    model_set = set(models)
    for model in models:
        brain.validate_model_id(model)
    parent_checkpoints = exact_map(parent_audit.get("checkpoint_sha256_by_model"), model_set, "parent checkpoint coverage")
    for model, expected in audit["AUDITED_CHECKPOINT_SHA256"].items():
        if parent_checkpoints.get(model) != list(expected):
            raise brain.BrainReferenceValidationError("Standardization parent changed an original audited checkpoint.")
    if (set(parent_brain.get("models", {})) != model_set
            or set(parent_atlas.get("models", {})) != model_set
            or parent_audit.get("voxel_order_sha256") != audit["AUDITED_VOXEL_ORDER_SHA256"]):
        raise brain.BrainReferenceValidationError("Standardization parent spatial identities are incomplete.")
    # Root release verification checks every rollback file. Independently bind
    # the spatial trust anchors and unchanged anatomy here as well.
    snapshot_files = snapshot.get("files")
    if not isinstance(snapshot_files, dict):
        raise brain.BrainReferenceValidationError("Standardization parent lacks its recoverable snapshot index.")
    anchors = {
        **ANATOMY_SHA256,
        "artifacts/reference/manifest.json": PARENT_MANIFEST_SHA256,
        "public/data/brain/reference/manifest.json": PARENT_BRAIN_MANIFEST_SHA256,
        "public/data/brain/reference/spatial-audit.json": PARENT_SPATIAL_AUDIT_SHA256,
        "public/data/voxel_selectivity_atlas/manifest.json": PARENT_ATLAS_MANIFEST_SHA256,
    }
    if any(snapshot_files.get(path) != expected for path, expected in anchors.items()):
        raise brain.BrainReferenceValidationError("Standardization snapshot index changed a reviewed spatial anchor.")
    for relative, expected in snapshot_files.items():
        if relative.startswith("public/data/brain/") and not relative.startswith("public/data/brain/reference/"):
            path = (baseline_root / relative).resolve()
            if not path.is_relative_to(baseline_root.resolve()) or brain.sha256(path.read_bytes()) != expected:
                raise brain.BrainReferenceValidationError("Standardization parent anatomy or voxel mapper changed.")

    reference = json.loads((artifact_root / "manifest.json").read_bytes())
    version = reference.get("reference_version")
    if (reference.get("dataset") != "nsd_1000" or reference.get("image_count") != brain.REFERENCE_COUNT
            or reference.get("reference_count") != brain.REFERENCE_COUNT or reference.get("voxel_counts") != brain.FULL_COUNTS
            or reference.get("image_ids_sha256") != brain.canonical_json_sha256(brain.canonical_image_ids())
            or not isinstance(version, str) or re.fullmatch(r"[a-f0-9]{64}", version) is None
            or version in {PARENT_REFERENCE_VERSION, brain.EXPECTED_REFERENCE_VERSION}
            or reference.get("checksum") != version
            or reference.get("supported_models") != sorted(models)
            or "coverage_extension" in reference):
        raise brain.BrainReferenceValidationError("Standardization must be a distinct complete release, not an append or relabeled parent.")
    exact_map(reference.get("artifacts"), model_set, "reference artifacts")
    source_selection = exact_map(reference.get("source_selection"), {f"{m}/{r}" for m in models for r in brain.ROIS}, "ROI source selections")
    lineage = reference.get("prediction_standardization")
    if (not isinstance(lineage, dict) or lineage.get("policy") != POLICY
            or lineage.get("analysis_input_policy") != ANALYSIS_INPUT_POLICY
            or lineage.get("parent_reference_version") != PARENT_REFERENCE_VERSION
            or lineage.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
            or lineage.get("parent_snapshot_sha256") != brain.sha256(snapshot_bytes)
            or lineage.get("regenerated_models") != sorted(models)):
        raise brain.BrainReferenceValidationError("Standardization lacks exact full-roster regeneration and rollback lineage.")
    proofs = exact_map(lineage.get("inference_provenance_sha256"), model_set, "inference provenance")
    precision_policies = exact_map(lineage.get("prediction_policies"), model_set, "numeric policies")
    prediction_policies = exact_map(lineage.get("model_prediction_policies"), model_set, "prediction/noise policies")
    core_root = audit_path.parents[2]
    layer_tree = ast.parse((audit_path.parents[1] / "constants/best_readout_layers.py").read_text())
    layers = next((ast.literal_eval(n.value) for n in layer_tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "best_readout_layers" for t in n.targets)), None)
    if not isinstance(layers, dict):
        raise brain.BrainReferenceValidationError("Canonical NSD readout metadata is unavailable.")
    precision_path = core_root / "cortex/utils/inference_precision.py"
    precision_hash = brain.sha256(precision_path.read_bytes())
    expected_images = [{"id": image, "sha256": brain.sha256((image_root / image).read_bytes())} for image in brain.canonical_image_ids()]
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for model in sorted(models):
        model_root = (raw_root / model).resolve()
        if not model_root.is_relative_to(raw_root.resolve()):
            raise brain.BrainReferenceValidationError("Standardized raw model path escapes its root.")
        proof_bytes = (model_root / "provenance.json").read_bytes()
        proof = json.loads(proof_bytes)
        numeric_policy = proof.get("numeric_policy")
        source_hashes = proof.get("core_inference_source_sha256")
        declared_precision = precision_policies[model]
        if (not isinstance(numeric_policy, dict) or numeric_policy != brain.INFERENCE_NUMERIC_POLICY
                or any(type(numeric_policy.get(k)) is not type(v) for k, v in brain.INFERENCE_NUMERIC_POLICY.items())
                or proof.get("precision_helper_sha256") != precision_hash
                or not isinstance(source_hashes, dict)
                or source_hashes.get("cortex/utils/inference_precision.py") != precision_hash
                or not isinstance(declared_precision, dict)
                or declared_precision != {"numeric_policy": brain.INFERENCE_NUMERIC_POLICY, "precision_helper_sha256": precision_hash}
                or any(type(declared_precision["numeric_policy"].get(k)) is not type(v) for k, v in brain.INFERENCE_NUMERIC_POLICY.items())):
            raise brain.BrainReferenceValidationError(f"{model}: all-model FP32 policy/helper identity is not verified.")
        policy = brain.expected_prediction_policy(model)
        if (brain.sha256(proof_bytes) != proofs[model] or proof.get("schema_version") != 1
                or proof.get("model") != model or proof.get("dataset") != "nsd_1000"
                or proof.get("reference_count") != brain.REFERENCE_COUNT
                or proof.get("image_ids") != brain.canonical_image_ids() or proof.get("images") != expected_images
                or proof.get("image_set_sha256") != brain.canonical_json_sha256(expected_images)
                or proof.get("subject_order") != ["s1", "s2", "s5", "s7"]
                or proof.get("subject_voxel_counts") != brain.SUBJECT_COUNTS or proof.get("voxel_counts") != brain.FULL_COUNTS
                or proof.get("preprocessing") != brain.INFERENCE_PREPROCESSING or proof.get("seed") != 0
                or not isinstance(proof.get("backbone_state_sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", proof["backbone_state_sha256"]) is None
                or any(proof.get(k) != v or type(proof.get(k)) is not type(v) for k, v in policy.items())
                or not isinstance(prediction_policies[model], dict) or prediction_policies[model] != policy
                or any(type(prediction_policies[model].get(k)) is not type(v) for k, v in policy.items())
                or set(proof.get("rois", {})) != set(brain.ROIS)):
            raise brain.BrainReferenceValidationError(f"{model}: fresh inference does not establish canonical image/readout/noise identity.")
        # All source fingerprints must describe the actual frozen code, not
        # copied provenance from a previous helper or constructor revision.
        for relative, digest in source_hashes.items():
            path = (core_root / relative).resolve()
            if (not path.is_relative_to(core_root.resolve()) or not path.is_file()
                    or brain.sha256(path.read_bytes()) != digest):
                raise brain.BrainReferenceValidationError(f"{model}: inference source fingerprint changed: {relative}.")
        matrices[model] = {}
        for roi_index, roi in enumerate(brain.ROIS):
            record = proof["rois"][roi]
            entry = reference["artifacts"][model][roi]
            checkpoint = Path(record.get("checkpoint_path", ""))
            layer = layers.get("nsd_1000", {}).get(roi, {}).get(model)
            columns = record.get("voxel_columns")
            if (layer is None or record.get("readout_layer") != layer or not checkpoint.is_absolute()
                    or checkpoint.name != f"{model}_nsd_1000_all_{roi}_{layer}.model" or not checkpoint.is_file()
                    or brain.sha256(checkpoint.read_bytes()) != record.get("checkpoint_sha256")
                    or record.get("checkpoint_sha256") != parent_checkpoints[model][roi_index]
                    or record.get("path") != entry.get("source_file") or record.get("sha256") != entry.get("source_sha256")
                    or record.get("output_shape") != [brain.REFERENCE_COUNT, brain.FULL_COUNTS[roi]]
                    or not isinstance(columns, list) or len(columns) != brain.FULL_COUNTS[roi]
                    or brain.canonical_json_sha256(columns) != audit["AUDITED_VOXEL_ORDER_SHA256"][roi]
                    or record.get("voxel_order_sha256") != audit["AUDITED_VOXEL_ORDER_SHA256"][roi]
                    or source_selection[f"{model}/{roi}"].get("selected") != entry["source_file"]):
                raise brain.BrainReferenceValidationError(f"{model}/{roi}: standardized checkpoint or full voxel order changed.")
            matrices[model][roi] = brain.validated_roi_matrix(artifact_root, model_root, entry, roi, audit["AUDITED_VOXEL_ORDER_SHA256"][roi])
    contract = {
        "schema_version": 2, "policy": POLICY,
        "analysis_input_policy": ANALYSIS_INPUT_POLICY,
        "baseline_reference_version": brain.EXPECTED_REFERENCE_VERSION,
        "parent_reference_version": PARENT_REFERENCE_VERSION,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_snapshot_sha256": brain.sha256(snapshot_bytes),
        "parent_spatial_audit_sha256": PARENT_SPATIAL_AUDIT_SHA256,
        "reference_version": version, "dataset": "nsd_1000", "subject_id": "s1",
        "response_space": brain.RESPONSE_SPACE,
        "subject_column_policy": "audited_s1_prefix_from_pooled_nsd_readout",
        "image_ids_sha256": brain.canonical_json_sha256(brain.canonical_image_ids()),
        "full_voxel_counts": brain.FULL_COUNTS, "subject1_voxel_counts": brain.COUNTS,
        "voxel_order_sha256": audit["AUDITED_VOXEL_ORDER_SHA256"],
        "checkpoint_sha256_by_model": parent_checkpoints,
        "inference_provenance_sha256_by_model": proofs,
        "model_prediction_policies": prediction_policies,
        "all_inference_numeric_policy": brain.INFERENCE_NUMERIC_POLICY,
        "precision_helper_sha256": precision_hash,
    }
    return reference, contract, matrices
