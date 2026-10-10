from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.data.generate_model_reference_predictions import (
    DATASET, IMAGE_IDS, PREPROCESSING, ROIS, SUBJECT_ORDER, SUBJECT_VOXEL_COUNTS,
    VOXEL_COUNTS, GenerationValidationError, canonical_json_sha256,
    model_inventory, offline_network_guard, ordered_voxel_columns, read_verified_csv, prepare_backbone_overlay,
    select_models, sha256_file, validate_predictions, verify_completed_triad, write_prediction_csv,
    validate_parallel_execution,
    NUMERIC_POLICY, verify_precision_provenance,
    validated_output_root,
    failure_output_root,
)


class ReferencePredictionGenerationTests(unittest.TestCase):
    def test_canonical_headers_match_independently_audited_archived_order(self):
        expected = {
            "ffa": "23e76a83c1f012119e805c45b79206022932ec0b2b3583bc08be322a2fdeb40a",
            "ppa": "ef9cde82b2838d705fce29012f88553baf9b602b383e315b11f91f4a04a9d044",
            "eba": "38f6fbbe08a48e874c6281f0b904e8f68d53e92aebe3fba19553a1a3e43deaf6",
        }
        for roi in ROIS:
            columns = ordered_voxel_columns(roi)
            self.assertEqual(len(columns), VOXEL_COUNTS[roi])
            self.assertEqual(columns[:SUBJECT_VOXEL_COUNTS["s1"][roi]], [f"s1_{i}" for i in range(SUBJECT_VOXEL_COUNTS["s1"][roi])])
            self.assertEqual(canonical_json_sha256(columns), expected[roi])

    def test_roster_excludes_ceiling_and_does_not_require_backbone_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cortex/constants").mkdir(parents=True)
            (root / "cortex/models").mkdir()
            layers = {DATASET: {r: {"existing": "layer", "missing": "layer"} for r in ROIS}}
            (root / "cortex/constants/best_readout_layers.py").write_text("best_readout_layers = " + repr(layers))
            (root / "cortex/models/__init__.py").write_text("backbone_models = {Existing: {'name': 'existing'}, Missing: {'name': 'missing'}}")
            scores = root / "scores.json"
            scores.write_text(json.dumps({r: {"existing": 0.5, "missing": 0.4, "ceiling": 1.0} for r in ROIS}))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"supported_models": ["existing"]}))
            (root / DATASET).mkdir()
            for roi in ROIS:
                (root / DATASET / f"missing_{DATASET}_all_{roi}_layer.model").write_bytes(b"trusted checkpoint fixture")
            inventory = model_inventory(root, root, scores, manifest)
            self.assertEqual(set(inventory), {"existing", "missing"})
            self.assertEqual(select_models(inventory, None, True), ["missing"])
            self.assertEqual(select_models(inventory, None, False, True), ["existing", "missing"])
            self.assertTrue(all(e["readable"] for e in inventory["missing"]["rois"].values()))
            self.assertFalse(inventory["existing"]["rois"]["ffa"]["readable"])

    def test_unknown_or_ambiguous_model_selection_is_rejected(self):
        inventory = {"model": {"existing_reference": False}}
        for models, all_missing in ((None, False), (["model"], True), (["ceiling"], False)):
            with self.subTest(models=models, all_missing=all_missing), self.assertRaises(GenerationValidationError):
                select_models(inventory, models, all_missing)

    def test_all_scored_selection_is_independent_of_the_current_reference_roster(self):
        inventory = {"previously_added": {"existing_reference": True},
                     "original": {"existing_reference": True}}
        self.assertEqual(select_models(inventory, None, False, True), ["original", "previously_added"])
        for models, missing in ((["original"], False), (None, True)):
            with self.assertRaises(GenerationValidationError):
                select_models(inventory, models, missing, True)

    def test_staging_outputs_allow_isolated_standardization_but_reject_live_or_broad_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.data.generate_model_reference_predictions.REPO_ROOT", root):
                for name in ("coverage-expansion", "precision-standardization"):
                    staged = root / "artifacts" / name / "raw-fp32"
                    self.assertEqual(validated_output_root(staged), staged.resolve())
                    with self.assertRaises(GenerationValidationError):
                        validated_output_root(staged.parent)
                for unsafe in (root, root / "artifacts", root / "data/image_by_model", root / "public/data"):
                    with self.assertRaises(GenerationValidationError):
                        validated_output_root(unsafe)

    def test_failure_forensics_are_outside_the_canonical_model_inventory(self):
        raw = Path("/fixture/artifacts/precision-standardization/raw-fp32")
        self.assertEqual(failure_output_root(raw), raw.parent / "failures/raw-fp32")
        self.assertFalse(failure_output_root(raw).is_relative_to(raw))

    def test_invalid_shapes_types_and_nonfinite_predictions_are_rejected(self):
        for matrix in (np.zeros((185, 715), dtype=np.float32), np.zeros((185, 716), dtype=np.float64), np.full((185, 716), np.nan, dtype=np.float32)):
            with self.assertRaises(GenerationValidationError):
                validate_predictions(matrix, "ffa")

    def test_csv_preserves_full_float32_values_and_exact_image_order(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(185, 716)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_prediction_csv(path, "ffa", matrix)
            np.testing.assert_array_equal(read_verified_csv(path, "ffa"), matrix)
            text = path.read_text()
            path.write_text(text.replace("image_001.png", "image_002.png", 1))
            with self.assertRaisesRegex(GenerationValidationError, "image order"):
                read_verified_csv(path, "ffa")

    def test_resume_requires_all_three_verified_checkpoints_sources_and_image_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [{"id": i, "sha256": "a" * 64} for i in IMAGE_IDS]
            entry = {"rois": {}}
            provenance = {
                "schema_version": 1, "model": "fixture", "dataset": DATASET,
                "image_ids": list(IMAGE_IDS), "images": images, "image_set_sha256": canonical_json_sha256(images),
                "subject_order": list(SUBJECT_ORDER), "subject_voxel_counts": SUBJECT_VOXEL_COUNTS,
                "voxel_counts": VOXEL_COUNTS, "preprocessing": PREPROCESSING, "rois": {},
            }
            for roi in ROIS:
                checkpoint = root / f"{roi}.model"
                checkpoint.write_bytes((roi + " checkpoint").encode())
                name = f"murtylab_fixture_{DATASET}_{roi}_20261010000000000Z.csv"
                path = root / name
                write_prediction_csv(path, roi, np.zeros((185, VOXEL_COUNTS[roi]), dtype=np.float32))
                entry["rois"][roi] = {"checkpoint_path": str(checkpoint), "readout_layer": "model.layer"}
                columns = ordered_voxel_columns(roi)
                provenance["rois"][roi] = {
                    "path": name, "sha256": sha256_file(path), "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint), "readout_layer": "model.layer",
                    "output_shape": [185, VOXEL_COUNTS[roi]], "voxel_columns": columns,
                    "voxel_order_sha256": canonical_json_sha256(columns),
                }
            (root / "provenance.json").write_text(json.dumps(provenance))
            self.assertEqual(verify_completed_triad(root, "fixture", entry, images)["model"], "fixture")
            changed_images = [{"id": i, "sha256": "b" * 64} for i in IMAGE_IDS]
            with self.assertRaises(GenerationValidationError):
                verify_completed_triad(root, "fixture", entry, changed_images)
            (root / "ffa.model").write_bytes(b"different readout")
            with self.assertRaisesRegex(GenerationValidationError, "checkpoint provenance"):
                verify_completed_triad(root, "fixture", entry, images)

    def test_network_is_blocked_and_original_request_functions_are_restored(self):
        import urllib.request
        original = urllib.request.urlopen
        with offline_network_guard(True):
            with self.assertRaisesRegex(GenerationValidationError, "Network request blocked"):
                urllib.request.urlopen("https://example.invalid/backbone")
        self.assertIs(urllib.request.urlopen, original)

    def test_offline_hub_lookup_preserves_hugging_face_local_cache_fallback(self):
        import os
        import requests
        from huggingface_hub.errors import OfflineModeIsEnabled
        original = os.system
        with offline_network_guard(True):
            with self.assertRaises(OfflineModeIsEnabled):
                requests.get("https://example.invalid/cache-metadata")
            with self.assertRaisesRegex(GenerationValidationError, "Network request blocked"):
                os.system("wget https://example.invalid/weights")
        self.assertIs(os.system, original)

    def test_torch_hub_captured_urlopen_alias_is_also_blocked(self):
        import torch.hub
        from urllib.error import URLError
        original = torch.hub.urlopen
        with offline_network_guard(True):
            with self.assertRaisesRegex(URLError, "Network request blocked") as blocked:
                torch.hub.urlopen("https://user:password@example.invalid/weights?token=secret#private")
            self.assertNotIn("password", str(blocked.exception))
            self.assertNotIn("token", str(blocked.exception))
        self.assertIs(torch.hub.urlopen, original)

    def test_torch_hub_offline_default_branch_probe_uses_existing_checkout(self):
        import torch.hub
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "facebookresearch_dinov2_main").mkdir()
            with patch("torch.hub.get_dir", return_value=directory), offline_network_guard(True):
                self.assertEqual(torch.hub._parse_repo_info("facebookresearch/dinov2"),
                                 ("facebookresearch", "dinov2", "main"))
            with patch("torch.hub.get_dir", return_value=directory), offline_network_guard(True):
                with self.assertRaisesRegex(RuntimeError, "no internet connection"):
                    torch.hub._parse_repo_info("uncached/repository")

    def test_dataparallel_readout_hooks_require_one_visible_cuda_device(self):
        import torch
        backbone = torch.nn.DataParallel(torch.nn.Identity())
        with patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(GenerationValidationError, "CUDA_VISIBLE_DEVICES=0"):
                validate_parallel_execution(backbone, "cuda:0")
            validate_parallel_execution(torch.nn.Identity(), "cuda:0")
            validate_parallel_execution(backbone, "cpu")
        with patch("torch.cuda.device_count", return_value=1):
            validate_parallel_execution(backbone, "cuda:0")

    def test_fp32_resume_requires_actual_policy_and_precision_helper_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            core = Path(directory)
            helper = core / "cortex/utils/inference_precision.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("trusted precision helper")
            checksum = sha256_file(helper)
            record = {"numeric_policy": NUMERIC_POLICY.copy(), "precision_helper_sha256": checksum,
                      "core_inference_source_sha256": {"cortex/utils/inference_precision.py": checksum}}
            verify_precision_provenance(record, core)
            with self.assertRaisesRegex(GenerationValidationError, "numeric policy"):
                verify_precision_provenance({}, core)
            record["numeric_policy"]["cudnn_allow_tf32"] = 0
            with self.assertRaisesRegex(GenerationValidationError, "numeric policy"):
                verify_precision_provenance(record, core)
            record["numeric_policy"] = NUMERIC_POLICY.copy()
            helper.write_text("changed helper")
            with self.assertRaisesRegex(GenerationValidationError, "source identity"):
                verify_precision_provenance(record, core)

    def test_overlay_links_readable_weights_but_copies_mutable_metadata_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shared"
            (source / "hub/checkpoints").mkdir(parents=True)
            (source / "models--fixture/refs").mkdir(parents=True)
            (source / "hub/checkpoints/weights.pth").write_bytes(b"large immutable weights")
            (source / "hub/trusted_list").write_text("fixture_repository\n")
            (source / "models--fixture/refs/main").write_text("commit\n")
            target = root / "artifacts/coverage-expansion/cache"
            with patch("scripts.data.generate_model_reference_predictions.REPO_ROOT", root):
                manifest = prepare_backbone_overlay(source, target)
            self.assertTrue((target / "hub/checkpoints/weights.pth").is_symlink())
            self.assertFalse((target / "hub").is_symlink())
            self.assertFalse((target / "hub/trusted_list").is_symlink())
            self.assertFalse((target / "models--fixture/refs/main").is_symlink())
            self.assertEqual(len(manifest["linked"]), 1)
            self.assertEqual(len(manifest["copied_metadata"]), 2)
            self.assertEqual((source / "hub/trusted_list").read_text(), "fixture_repository\n")


if __name__ == "__main__":
    unittest.main()
