from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import ast
import copy
import csv
import shutil
from pathlib import Path

import numpy as np

from scripts.data.build_brain_reference_artifacts import (
    COUNTS,
    EXPECTED_REFERENCE_VERSION,
    OFFSETS,
    REPO_ROOT,
    ROIS,
    ROW_WIDTH,
    BrainReferenceValidationError,
    canonical_image_ids,
    load_audit_constants,
    validated_roi_matrix,
    append_reference_brain_assets,
    canonical_json_sha256,
    validate_model_id,
    validate_spatial_expansion,
    FULL_COUNTS,
    SUBJECT_COUNTS,
    INFERENCE_PREPROCESSING,
    INFERENCE_NUMERIC_POLICY,
    expected_prediction_policy,
)


class BrainReferenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = REPO_ROOT / "public/data/brain/reference"
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())
        cls.source_manifest = json.loads((REPO_ROOT / "artifacts/reference/manifest.json").read_text())
        cls.audit = load_audit_constants(REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py")
        cls.checkpoints = cls.audit["AUDITED_CHECKPOINT_SHA256"]
        if (cls.output / "spatial-audit.json").is_file():
            cls.checkpoints = json.loads((cls.output / "spatial-audit.json").read_text())["checkpoint_sha256_by_model"]

    def test_public_reference_identity_is_raw_predicted_subject1_not_measured_or_cluster_space(self):
        manifest = self.manifest
        self.assertEqual(manifest["referenceVersion"], self.source_manifest["reference_version"])
        self.assertEqual(manifest["imageIds"], canonical_image_ids())
        self.assertEqual(manifest["dataset"], "nsd_1000")
        self.assertEqual(manifest["responseKind"], "model_prediction")
        self.assertEqual(manifest["subjectId"], "s1")
        self.assertEqual(manifest["normalization"], "none")
        self.assertEqual(manifest["dtype"], "float32-le")
        self.assertEqual(manifest["rowWidth"], ROW_WIDTH)
        self.assertEqual(set(manifest["models"]), set(self.source_manifest["supported_models"]))
        self.assertTrue(set(self.audit["AUDITED_CHECKPOINT_SHA256"]) <= set(manifest["models"]))

    def test_all_model_payloads_exactly_preserve_raw_subject1_prefixes_and_baseline_26_identity(self):
        for model, entry in self.manifest["models"].items():
            with self.subTest(model=model):
                payload = (self.output / entry["url"]).read_bytes()
                self.assertEqual(len(payload), 185 * ROW_WIDTH * 4)
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                matrix = np.frombuffer(payload, dtype="<f4").reshape(185, ROW_WIDTH)
                self.assertTrue(np.isfinite(matrix).all())
                for roi in ROIS:
                    source = self.source_manifest["artifacts"][model][roi]
                    with np.load(REPO_ROOT / "artifacts/reference" / source["path"], allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(
                            matrix[:, OFFSETS[roi]:OFFSETS[roi] + COUNTS[roi]],
                            arrays["reference_voxels"][:, :COUNTS[roi]],
                        )
                    provenance = entry["provenance"][roi]
                    self.assertEqual(provenance["artifactSha256"], source["sha256"])
                    self.assertEqual(provenance["sourceSha256"], source["source_sha256"])
                    self.assertEqual(provenance["voxelOrderSha256"], self.audit["AUDITED_VOXEL_ORDER_SHA256"][roi])
                    self.assertEqual(provenance["checkpointSha256"], self.checkpoints[model][ROIS.index(roi)])
                if model in self.audit["AUDITED_CHECKPOINT_SHA256"]:
                    baseline = REPO_ROOT / "artifacts/coverage-expansion/baseline/public/data/brain/reference"
                    if baseline.is_dir():
                        self.assertEqual(payload, (baseline / f"{model}.f32").read_bytes())

    def test_audit_reader_never_executes_source_code_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.py"
            path.write_text("raise RuntimeError('must not execute')\nAUDITED_REFERENCE_VERSION = 'wrong'\n")
            with self.assertRaises(BrainReferenceValidationError):
                load_audit_constants(path)

    def test_exporter_rejects_changed_source_or_artifact_checksums_and_voxel_order(self):
        root = REPO_ROOT / "artifacts/reference"
        raw = REPO_ROOT / "data/image_by_model"
        entry = self.source_manifest["artifacts"]["clip_rn50"]["ffa"]
        expected_order = self.audit["AUDITED_VOXEL_ORDER_SHA256"]["ffa"]
        for changed, order in (
            ({**entry, "sha256": "0" * 64}, expected_order),
            ({**entry, "source_sha256": "0" * 64}, expected_order),
            (entry, "0" * 64),
        ):
            with self.subTest(changed=changed.get("sha256"), order=order):
                with self.assertRaises(BrainReferenceValidationError):
                    validated_roi_matrix(root, raw, changed, "ffa", order)
        validated = validated_roi_matrix(root, raw, entry, "ffa", expected_order)
        self.assertEqual(validated.shape, (185, 156))
        self.assertEqual(validated.dtype, np.dtype("<f4"))

    def test_decimal_model_identity_is_safe_without_permitting_traversal(self):
        validate_model_id("toponets_rn18_tau_0.5")
        for model in ("../clip_rn50", "x/y", ".", "..", "x..5", "x.5/path", "x.", "X"):
            with self.subTest(model=model), self.assertRaises(BrainReferenceValidationError):
                validate_model_id(model)

    def test_vone_rn50_noise_free_checkpoint_is_not_labeled_stochastic(self):
        self.assertEqual(expected_prediction_policy("vone_rn50"), {
            "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": [],
        })
        for model in ("vone_alexnet", "vone_cornet_s"):
            with self.subTest(model=model):
                self.assertEqual(expected_prediction_policy(model), {
                    "response_policy": "seeded_stochastic_sampling", "deterministic_per_stimulus": False,
                    "noise_modes": ["neuronal"],
                })


class SpatialExpansionTests(unittest.TestCase):
    """Synthetic new inference records exercise lineage, not scientific predictions."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        snapshot = REPO_ROOT / "artifacts/coverage-expansion/baseline"
        cls.baseline_root = snapshot if (snapshot / "artifacts/reference/manifest.json").is_file() else REPO_ROOT
        cls.baseline_artifacts = cls.baseline_root / "artifacts/reference"
        cls.artifacts = cls.root / "artifacts"
        shutil.copytree(cls.baseline_artifacts, cls.artifacts)
        cls.raw_root = cls.root / "raw"
        cls.model = "dinov2"
        model_root = cls.raw_root / cls.model
        model_root.mkdir(parents=True)
        cls.audit_path = REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py"
        audit = load_audit_constants(cls.audit_path)
        baseline = json.loads((cls.baseline_artifacts / "manifest.json").read_text())
        cls.reference = copy.deepcopy(baseline)
        cls.reference["reference_version"] = "a" * 64
        cls.reference["checksum"] = "a" * 64
        cls.reference["supported_models"] = sorted([*baseline["supported_models"], cls.model])
        cls.reference["artifacts"][cls.model] = {}
        layer_tree = ast.parse((cls.audit_path.parents[1] / "constants/best_readout_layers.py").read_text())
        layers = next(ast.literal_eval(node.value) for node in layer_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "best_readout_layers" for target in node.targets))
        images = [{"id": image, "sha256": hashlib.sha256((REPO_ROOT / "public/images" / image).read_bytes()).hexdigest()} for image in canonical_image_ids()]
        cls.proof = {
            "schema_version": 1, "model": cls.model, "dataset": "nsd_1000", "reference_count": 185,
            "image_ids": canonical_image_ids(), "images": images, "image_set_sha256": canonical_json_sha256(images),
            "subject_order": ["s1", "s2", "s5", "s7"], "subject_voxel_counts": SUBJECT_COUNTS,
            "voxel_counts": FULL_COUNTS, "preprocessing": INFERENCE_PREPROCESSING, "seed": 0,
            "backbone_state_sha256": "b" * 64, "rois": {},
            "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": [],
            "numeric_policy": INFERENCE_NUMERIC_POLICY,
            "precision_helper_sha256": hashlib.sha256((cls.audit_path.parents[1] / "utils/inference_precision.py").read_bytes()).hexdigest(),
        }
        cls.proof["core_inference_source_sha256"] = {
            "cortex/utils/inference_precision.py": cls.proof["precision_helper_sha256"],
        }
        for roi in ROIS:
            source_entry = baseline["artifacts"]["clip_rn50"][roi]
            source_name = f"murtylab_{cls.model}_nsd_1000_{roi}_20261010000000000Z.csv"
            raw_path = model_root / source_name
            shutil.copyfile(cls.baseline_root / "data/image_by_model" / source_entry["source_file"], raw_path)
            artifact_relative = f"models/{cls.model}/{roi}.npz"
            artifact_path = cls.artifacts / artifact_relative
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cls.baseline_artifacts / source_entry["path"], artifact_path)
            cls.reference["artifacts"][cls.model][roi] = {**source_entry, "path": artifact_relative, "source_file": source_name}
            cls.reference["source_selection"][f"{cls.model}/{roi}"] = {"selected": source_name}
            layer = layers["nsd_1000"][roi][cls.model]
            checkpoint = cls.root / f"{cls.model}_nsd_1000_all_{roi}_{layer}.model"
            checkpoint.write_bytes(f"synthetic checkpoint fixture {roi}".encode())
            with raw_path.open(newline="") as handle:
                columns = next(csv.reader(handle))[1:]
            cls.proof["rois"][roi] = {
                "path": source_name, "sha256": source_entry["source_sha256"],
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "readout_layer": layer, "output_shape": [185, FULL_COUNTS[roi]],
                "voxel_columns": columns, "voxel_order_sha256": audit["AUDITED_VOXEL_ORDER_SHA256"][roi],
            }
        cls.reference["coverage_extension"] = {
            "policy": "append_only_preserve_baseline_artifacts_v1",
            "parent_reference_version": EXPECTED_REFERENCE_VERSION,
            "baseline_models": baseline["supported_models"], "added_models": [cls.model],
            "new_inference_provenance_sha256": {},
            "prediction_policies": {cls.model: {
                "numeric_policy": INFERENCE_NUMERIC_POLICY,
                "precision_helper_sha256": cls.proof["precision_helper_sha256"],
            }},
        }
        cls.restore_fixture()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def restore_fixture(cls, proof=None, reference=None):
        proof = copy.deepcopy(cls.proof if proof is None else proof)
        reference = copy.deepcopy(cls.reference if reference is None else reference)
        payload = json.dumps(proof, sort_keys=True).encode()
        (cls.raw_root / cls.model / "provenance.json").write_bytes(payload)
        reference["coverage_extension"]["new_inference_provenance_sha256"][cls.model] = hashlib.sha256(payload).hexdigest()
        (cls.artifacts / "manifest.json").write_text(json.dumps(reference))

    def setUp(self):
        self.restore_fixture()

    def kwargs(self):
        return {"artifact_root": self.artifacts, "raw_root": self.raw_root,
                "baseline_artifact_root": self.baseline_artifacts, "audit_path": self.audit_path}

    def test_complete_extension_binds_checkpoint_order_and_preserves_old_brain_bytes(self):
        reference, contract, matrices = validate_spatial_expansion(**self.kwargs())
        self.assertEqual(set(matrices), {self.model})
        self.assertEqual(contract["reference_version"], reference["reference_version"])
        self.assertEqual(contract["new_inference_numeric_policy"], INFERENCE_NUMERIC_POLICY)
        self.assertEqual(contract["precision_helper_sha256"], self.proof["precision_helper_sha256"])
        self.assertEqual(len(contract["checkpoint_sha256_by_model"]), 27)
        self.assertEqual(matrices[self.model]["eba"].shape, (185, 518))
        output = self.root / "brain-output"
        try:
            manifest = append_reference_brain_assets(**self.kwargs(),
                baseline_brain_root=self.baseline_root / "public/data/brain/reference", output_root=output)
            self.assertEqual(len(manifest["models"]), 27)
            for model in set(manifest["models"]) - {self.model}:
                self.assertEqual((output / f"{model}.f32").read_bytes(), (self.baseline_root / "public/data/brain/reference" / f"{model}.f32").read_bytes())
            self.assertTrue((output / "spatial-audit.json").is_file())
            with self.assertRaisesRegex(BrainReferenceValidationError, "new staging"):
                append_reference_brain_assets(**self.kwargs(), baseline_brain_root=self.baseline_root / "public/data/brain/reference", output_root=output)
        finally:
            if output.exists():
                shutil.rmtree(output)

    def test_an_added_name_or_changed_baseline_record_cannot_become_audited(self):
        reference = copy.deepcopy(self.reference)
        reference["artifacts"]["clip_rn50"]["ffa"]["source_sha256"] = "0" * 64
        self.restore_fixture(reference=reference)
        with self.assertRaisesRegex(BrainReferenceValidationError, "baseline reference artifact"):
            validate_spatial_expansion(**self.kwargs())

    def test_new_spatial_predictions_require_exact_fp32_policy_and_helper_bytes(self):
        for kind in ("missing_policy", "tf32", "autocast", "wrong_flag_type", "helper", "helper_source"):
            proof = copy.deepcopy(self.proof)
            if kind == "missing_policy":
                proof.pop("numeric_policy")
            elif kind == "tf32":
                proof["numeric_policy"]["cuda_matmul_allow_tf32"] = True
            elif kind == "autocast":
                proof["numeric_policy"]["cuda_autocast_enabled"] = True
            elif kind == "wrong_flag_type":
                proof["numeric_policy"]["cuda_matmul_allow_tf32"] = 0
            elif kind == "helper_source":
                proof["core_inference_source_sha256"]["cortex/utils/inference_precision.py"] = "0" * 64
            else:
                proof["precision_helper_sha256"] = "0" * 64
            self.restore_fixture(proof=proof)
            with self.subTest(kind=kind), self.assertRaisesRegex(BrainReferenceValidationError, "FP32 policy/helper"):
                validate_spatial_expansion(**self.kwargs())

    def test_wrong_checkpoint_same_width_voxel_order_or_preprocessing_fails_closed(self):
        for kind in ("checkpoint", "order", "preprocessing", "images"):
            proof = copy.deepcopy(self.proof)
            if kind == "checkpoint":
                proof["rois"]["ppa"]["checkpoint_sha256"] = "0" * 64
            elif kind == "order":
                proof["rois"]["ffa"]["voxel_columns"][0:2] = ["s1_1", "s1_0"]
            elif kind == "preprocessing":
                proof["preprocessing"]["tail_policy"] = "pad_to_expected_width"
            else:
                proof["images"][0]["sha256"] = "0" * 64
                proof["image_set_sha256"] = canonical_json_sha256(proof["images"])
            self.restore_fixture(proof=proof)
            with self.subTest(kind=kind), self.assertRaises(BrainReferenceValidationError):
                validate_spatial_expansion(**self.kwargs())

    def test_atlas_append_fits_only_new_model_and_copies_all_baseline_children_verbatim(self):
        from unittest.mock import patch
        from scripts.data.build_selectivity_atlas_artifacts import AtlasFit, append_selectivity_atlas_assets
        fit = AtlasFit(
            z_profiles=np.zeros((185, 844)), selectivity_features=np.zeros((844, 35)),
            pca_explained_variance_ratio_sum=0.9, labels=np.arange(844) % 14,
            embedding=np.zeros((844, 3)), distances_to_centroid=np.zeros(844),
            silhouette_by_voxel=np.zeros(844), silhouette=0.0,
            split_half_stability={"iterations": 6, "randomSeed": 42, "adjustedRandIndices": [0.0] * 6,
                "meanAdjustedRandIndex": 0.0, "minAdjustedRandIndex": 0.0, "threshold": 0.7,
                "meanPassesThreshold": False, "allPassThreshold": False},
        )
        output = self.root / "atlas-output"
        baseline_root = self.baseline_root / "public/data/voxel_selectivity_atlas"
        try:
            with patch("scripts.data.build_selectivity_atlas_artifacts.fit_selectivity_atlas", return_value=fit) as fitting:
                manifest = append_selectivity_atlas_assets(**self.kwargs(), baseline_atlas_root=baseline_root, output_dir=output)
            fitting.assert_called_once()
            self.assertEqual(fitting.call_args.args[0].shape, (185, 844))
            self.assertEqual(len(manifest["supportedModels"]), 27)
            baseline = json.loads((baseline_root / "manifest.json").read_text())
            for model, entry in baseline["models"].items():
                self.assertEqual(manifest["models"][model], entry)
                for field in ("pointsUrl", "summaryUrl"):
                    self.assertEqual((output / entry[field]).read_bytes(), (baseline_root / entry[field]).read_bytes())
        finally:
            if output.exists():
                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
