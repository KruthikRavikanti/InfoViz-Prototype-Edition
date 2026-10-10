"""Synthetic lineage tests, not new scientific prediction generation."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.data import build_brain_reference_artifacts as brain
from scripts.data import spatial_standardization as standard
from scripts.data.build_selectivity_atlas_artifacts import (
    AtlasConfig, AtlasFit, SubjectRoiSource, standardize_selectivity_atlas_assets,
)


class SpatialStandardizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.baseline = cls.root / "baseline"
        cls.artifacts = cls.root / "artifacts"
        cls.raw = cls.root / "raw"
        cls.core = cls.root / "core"
        cls.audit_path = cls.core / "cortex/comparison/brain_map_audit.py"
        cls.audit_path.parent.mkdir(parents=True)
        actual_core = brain.REPO_ROOT.parent / "cortex-core"
        shutil.copyfile(actual_core / "cortex/comparison/brain_map_audit.py", cls.audit_path)
        layer_path = cls.core / "cortex/constants/best_readout_layers.py"
        layer_path.parent.mkdir(parents=True)
        shutil.copyfile(actual_core / "cortex/constants/best_readout_layers.py", layer_path)
        layer_tree = ast.parse(layer_path.read_text())
        layers = next(ast.literal_eval(n.value) for n in layer_tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "best_readout_layers" for t in n.targets))
        helper = cls.core / "cortex/utils/inference_precision.py"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"synthetic frozen helper fixture, not executable inference\n")
        cls.helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
        cls.parent_audit = json.loads((actual_core / "cortex/comparison/brain_map_extensions" / f"{standard.PARENT_REFERENCE_VERSION}.json").read_bytes())
        cls.models = sorted(cls.parent_audit["checkpoint_sha256_by_model"])
        if len(cls.models) != 126:
            raise AssertionError("Fixture needs the reviewed full parent roster.")
        cls.parent_brain = json.loads((brain.REPO_ROOT / "public/data/brain/reference/manifest.json").read_bytes())
        cls.parent_atlas = json.loads((brain.REPO_ROOT / "public/data/voxel_selectivity_atlas/manifest.json").read_bytes())
        cls.parent_brain["referenceVersion"] = standard.PARENT_REFERENCE_VERSION
        cls.parent_atlas["referenceVersion"] = standard.PARENT_REFERENCE_VERSION
        # Pin-anchor reads and fake checkpoint hashes are isolated fixtures;
        # production always verifies actual bytes. Everything else is real
        # validation, including all126 proof keys, columns, images and policies.
        cls.documents = {
            cls.baseline / "artifacts/reference/manifest.json": {"reference_version": standard.PARENT_REFERENCE_VERSION, "supported_models": cls.models},
            cls.baseline / "public/data/brain/reference/manifest.json": cls.parent_brain,
            cls.baseline / "public/data/brain/reference/spatial-audit.json": cls.parent_audit,
            cls.baseline / "public/data/voxel_selectivity_atlas/manifest.json": cls.parent_atlas,
        }
        cls.sha_overrides = {}
        for path, document in cls.documents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document))
        cls.snapshot = {
            "reference_version": standard.PARENT_REFERENCE_VERSION, "models": cls.models,
            "files": {
                **standard.ANATOMY_SHA256,
                "artifacts/reference/manifest.json": standard.PARENT_MANIFEST_SHA256,
                "public/data/brain/reference/manifest.json": standard.PARENT_BRAIN_MANIFEST_SHA256,
                "public/data/brain/reference/spatial-audit.json": standard.PARENT_SPATIAL_AUDIT_SHA256,
                "public/data/voxel_selectivity_atlas/manifest.json": standard.PARENT_ATLAS_MANIFEST_SHA256,
            },
        }
        for relative, digest in standard.ANATOMY_SHA256.items():
            path = cls.baseline / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            cls.sha_overrides[path.read_bytes()] = digest
        (cls.baseline / "snapshot.json").write_text(json.dumps(cls.snapshot, sort_keys=True))
        for folder in ("brain/reference", "voxel_selectivity_atlas"):
            (cls.baseline / "public/data" / folder / "unrelated-note.txt").write_text("preserve this unrelated file\n")
        (cls.baseline / "public/data/brain/reference/clip_rn50.f32").write_bytes(b"historical payload must be refreshed")
        cls.reference = {
            "dataset": "nsd_1000", "image_count": 185, "reference_count": 185,
            "voxel_counts": brain.FULL_COUNTS, "image_ids_sha256": brain.canonical_json_sha256(brain.canonical_image_ids()),
            "reference_version": "f" * 64, "checksum": "f" * 64, "supported_models": cls.models,
            "artifacts": {}, "source_selection": {},
            "prediction_standardization": {
                "policy": standard.POLICY, "parent_reference_version": standard.PARENT_REFERENCE_VERSION,
                "analysis_input_policy": standard.ANALYSIS_INPUT_POLICY,
                "parent_manifest_sha256": standard.PARENT_MANIFEST_SHA256,
                "parent_snapshot_sha256": hashlib.sha256((cls.baseline / "snapshot.json").read_bytes()).hexdigest(),
                "regenerated_models": cls.models, "inference_provenance_sha256": {},
                "prediction_policies": {}, "model_prediction_policies": {},
            },
        }
        images = [{"id": x, "sha256": hashlib.sha256((brain.REPO_ROOT / "public/images" / x).read_bytes()).hexdigest()} for x in brain.canonical_image_ids()]
        cls.proofs = {}
        cls.matrices = {roi: np.full((185, brain.COUNTS[roi]), index + 0.25, dtype=np.float32) for index, roi in enumerate(brain.ROIS)}
        audit = brain.load_audit_constants(cls.audit_path)
        for model in cls.models:
            cls.reference["artifacts"][model] = {}
            policy = brain.expected_prediction_policy(model)
            proof = {
                "schema_version": 1, "model": model, "dataset": "nsd_1000", "reference_count": 185,
                "image_ids": brain.canonical_image_ids(), "images": images,
                "image_set_sha256": brain.canonical_json_sha256(images), "subject_order": ["s1", "s2", "s5", "s7"],
                "subject_voxel_counts": brain.SUBJECT_COUNTS, "voxel_counts": brain.FULL_COUNTS,
                "preprocessing": brain.INFERENCE_PREPROCESSING, "seed": 0, "backbone_state_sha256": "b" * 64,
                **policy, "numeric_policy": brain.INFERENCE_NUMERIC_POLICY,
                "precision_helper_sha256": cls.helper_hash,
                "core_inference_source_sha256": {"cortex/utils/inference_precision.py": cls.helper_hash}, "rois": {},
            }
            for i, roi in enumerate(brain.ROIS):
                layer = layers["nsd_1000"][roi][model]
                checkpoint = cls.root / "checkpoints" / f"{model}_nsd_1000_all_{roi}_{layer}.model"
                checkpoint.parent.mkdir(exist_ok=True)
                checkpoint.write_bytes(f"synthetic checkpoint {model}/{roi}".encode())
                digest = cls.parent_audit["checkpoint_sha256_by_model"][model][i]
                cls.sha_overrides[checkpoint.read_bytes()] = digest
                filename = f"murtylab_{model}_nsd_1000_{roi}_20261011000000000Z.csv"
                cls.reference["artifacts"][model][roi] = {"path": f"models/{model}/{roi}.npz", "sha256": "c" * 64, "source_file": filename, "source_sha256": "d" * 64}
                cls.reference["source_selection"][f"{model}/{roi}"] = {"selected": filename}
                columns = [f"{s}_{v}" for s in ("s1", "s2", "s5", "s7") for v in range(brain.SUBJECT_COUNTS[s][roi])]
                proof["rois"][roi] = {"path": filename, "sha256": "d" * 64, "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest, "readout_layer": layer, "output_shape": [185, brain.FULL_COUNTS[roi]],
                    "voxel_columns": columns, "voxel_order_sha256": audit["AUDITED_VOXEL_ORDER_SHA256"][roi]}
            cls.proofs[model] = copy.deepcopy(proof)
            cls.reference["prediction_standardization"]["prediction_policies"][model] = {"numeric_policy": brain.INFERENCE_NUMERIC_POLICY, "precision_helper_sha256": cls.helper_hash}
            cls.reference["prediction_standardization"]["model_prediction_policies"][model] = policy
        cls.artifacts.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.reference = copy.deepcopy(self.__class__.reference)
        self.proofs = copy.deepcopy(self.__class__.proofs)
        self.restore()
        self.doc_patch = patch.object(standard, "checked_document", side_effect=lambda path, digest: copy.deepcopy(self.documents[path]))
        self.sha_patch = patch.object(brain, "sha256", side_effect=lambda payload: self.sha_overrides.get(payload, hashlib.sha256(payload).hexdigest()))
        self.matrix_patch = patch.object(brain, "validated_roi_matrix", side_effect=lambda root, raw, entry, roi, digest: self.matrices[roi])
        for p in (self.doc_patch, self.sha_patch, self.matrix_patch):
            p.start()
            self.addCleanup(p.stop)

    def restore(self):
        for model, proof in self.proofs.items():
            path = self.raw / model / "provenance.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(proof, sort_keys=True).encode()
            path.write_bytes(payload)
            self.reference["prediction_standardization"]["inference_provenance_sha256"][model] = hashlib.sha256(payload).hexdigest()
        (self.artifacts / "manifest.json").write_text(json.dumps(self.reference))

    def kwargs(self):
        return {"artifact_root": self.artifacts, "raw_root": self.raw, "baseline_root": self.baseline, "audit_path": self.audit_path}

    def test_all_126_including_original_26_require_fresh_precision_proofs(self):
        reference, contract, matrices = standard.validate_spatial_standardization(**self.kwargs())
        self.assertEqual(len(matrices), 126)
        self.assertEqual(set(contract["inference_provenance_sha256_by_model"]), set(self.models))
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(contract["analysis_input_policy"], standard.ANALYSIS_INPUT_POLICY)
        self.assertEqual(contract["all_inference_numeric_policy"], brain.INFERENCE_NUMERIC_POLICY)
        self.assertNotIn("new_inference_numeric_policy", contract)
        self.assertNotIn("coverage_extension", reference)

    def test_old_26_proofs_cannot_be_missing_or_relabelled(self):
        for kind in ("missing_numeric", "tf32", "helper", "helper_source", "column_order", "checkpoint", "image", "noise", "noise_type"):
            self.proofs = copy.deepcopy(self.__class__.proofs)
            proof = self.proofs["clip_rn50"]
            if kind == "missing_numeric": proof.pop("numeric_policy")
            elif kind == "tf32": proof["numeric_policy"]["cudnn_allow_tf32"] = True
            elif kind == "helper": proof["precision_helper_sha256"] = "0" * 64
            elif kind == "helper_source": proof["core_inference_source_sha256"]["cortex/utils/inference_precision.py"] = "0" * 64
            elif kind == "column_order": proof["rois"]["ffa"]["voxel_columns"][0:2] = ["s1_1", "s1_0"]
            elif kind == "checkpoint": proof["rois"]["ffa"]["checkpoint_sha256"] = "0" * 64
            elif kind == "image": proof["images"][0]["sha256"] = "0" * 64
            elif kind == "noise": proof["noise_modes"] = ["neuronal"]
            else: proof["deterministic_per_stimulus"] = 1
            self.restore()
            with self.subTest(kind=kind), self.assertRaises(brain.BrainReferenceValidationError):
                standard.validate_spatial_standardization(**self.kwargs())

    def test_partial_lineage_and_append_claims_fail_closed(self):
        for kind in ("old_proof", "extra_model", "parent", "snapshot", "analysis_input", "missing_analysis_input", "append"):
            self.reference = copy.deepcopy(self.__class__.reference)
            self.restore()
            if kind == "old_proof": self.reference["prediction_standardization"]["inference_provenance_sha256"].pop("clip_rn50")
            elif kind == "extra_model": self.reference["supported_models"].append("ceiling")
            elif kind == "parent": self.reference["prediction_standardization"]["parent_manifest_sha256"] = "0" * 64
            elif kind == "snapshot": self.reference["prediction_standardization"]["parent_snapshot_sha256"] = "0" * 64
            elif kind == "analysis_input": self.reference["prediction_standardization"]["analysis_input_policy"] = "csv_float64"
            elif kind == "missing_analysis_input": self.reference["prediction_standardization"].pop("analysis_input_policy")
            else: self.reference["coverage_extension"] = {"policy": "append_only_preserve_baseline_artifacts_v1"}
            (self.artifacts / "manifest.json").write_text(json.dumps(self.reference))
            with self.subTest(kind=kind), self.assertRaises(brain.BrainReferenceValidationError):
                standard.validate_spatial_standardization(**self.kwargs())

    def test_brain_exports_every_fresh_matrix_and_preserves_only_unrelated_files(self):
        output = self.root / "new-brain"
        manifest = brain.standardize_reference_brain_assets(**self.kwargs(), output_root=output)
        self.addCleanup(shutil.rmtree, output)
        self.assertEqual(len(manifest["models"]), 126)
        expected = np.concatenate([self.matrices[r] for r in brain.ROIS], axis=1)
        actual = np.frombuffer((output / "clip_rn50.f32").read_bytes(), dtype="<f4").reshape(185, 844)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((output / "unrelated-note.txt").read_text(), "preserve this unrelated file\n")
        self.assertEqual(json.loads((output / "spatial-audit.json").read_text())["schema_version"], 2)
        with self.assertRaisesRegex(brain.BrainReferenceValidationError, "new staging"):
            brain.standardize_reference_brain_assets(**self.kwargs(), output_root=output)

    def test_atlas_refits_all_models_with_unchanged_method_and_preserves_unrelated_files(self):
        output = self.root / "new-atlas"
        fit = AtlasFit(z_profiles=np.zeros((185, 844)), selectivity_features=np.zeros((844, 35)),
            pca_explained_variance_ratio_sum=0.9, labels=np.arange(844) % 14, embedding=np.zeros((844, 3)),
            distances_to_centroid=np.zeros(844), silhouette_by_voxel=np.zeros(844), silhouette=0.0,
            split_half_stability={"iterations": 6, "randomSeed": 42, "adjustedRandIndices": [0.0] * 6,
                "meanAdjustedRandIndex": 0.0, "minAdjustedRandIndex": 0.0, "threshold": 0.7,
                "meanPassesThreshold": False, "allPassThreshold": False})
        def sources(*, model, **kwargs):
            return {roi: SubjectRoiSource(model=model, roi=roi, path=self.raw / model / self.reference["artifacts"][model][roi]["source_file"],
                sha256="d" * 64, image_ids=tuple(brain.canonical_image_ids()), responses=self.matrices[roi].astype(np.float64)) for roi in brain.ROIS}
        with patch("scripts.data.build_selectivity_atlas_artifacts.load_model_sources", side_effect=sources), \
                patch("scripts.data.build_selectivity_atlas_artifacts.fit_selectivity_atlas", return_value=fit) as fitting:
            manifest = standardize_selectivity_atlas_assets(**self.kwargs(), output_dir=output)
        self.addCleanup(shutil.rmtree, output)
        self.assertEqual(fitting.call_count, 126)
        self.assertEqual(manifest["method"], self.parent_atlas["method"])
        self.assertEqual((output / "unrelated-note.txt").read_text(), "preserve this unrelated file\n")
        self.assertEqual(len(manifest["models"]), 126)
        with self.assertRaisesRegex(ValueError, "method exactly"):
            standardize_selectivity_atlas_assets(**self.kwargs(), output_dir=self.root / "wrong-method", config=AtlasConfig(cluster_count=15))


class ParentBytePinTests(unittest.TestCase):
    def test_parent_json_requires_exact_bytes_not_a_copied_digest_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "parent.json"
            payload = b'{"reviewed":true}'
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(standard.checked_document(path, expected), {"reviewed": True})
            path.write_bytes(payload + b" ")
            with self.assertRaisesRegex(brain.BrainReferenceValidationError, "parent payload changed"):
                standard.checked_document(path, expected)


if __name__ == "__main__":
    unittest.main()
