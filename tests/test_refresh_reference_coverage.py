from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.data import refresh_reference_coverage as refresh
from scripts.data import verify_reference_refresh as verifier
from scripts.data import build_reference_artifacts as reference
import test_extend_reference_coverage as fixtures


class RefreshReferenceFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing complete full-width, 185-image fixture; its test
        # class itself is not imported into this module's discovery namespace.
        fixtures.ExpansionFixtureTests.setUpClass.__func__(cls)
        cls.baseline = cls.private.parents[1]
        shutil.copytree(cls.raw, cls.baseline / "data/image_by_model")
        refresh.expansion.write_json(cls.data / "murty185_zero_index.json", {
            roi: {"old": 0.5, "ceiling": 0.9} for roi in reference.ROIS
        })
        cls.core_root = cls.root / "core"
        cls.core = cls.core_root / "cortex/models/__init__.py"
        cls.core.parent.mkdir(parents=True)
        cls.core.write_text("backbone_models = {'old': {'name': 'old'}}\n")
        helper = cls.core_root / "cortex/utils/inference_precision.py"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"frozen uniform FP32 fixture helper")
        cls.source_hashes = {
            "cortex/utils/inference_precision.py": reference.sha256_file(helper),
            "cortex/models/__init__.py": reference.sha256_file(cls.core),
        }
        cls.images = cls.root / "images"
        cls.images.mkdir()
        for image in reference.canonical_image_ids():
            (cls.images / image).write_bytes(f"canonical fixture {image}".encode())
        image_records = [{"id": image, "sha256": reference.sha256_file(cls.images / image)}
                         for image in reference.canonical_image_ids()]
        cls.refresh_raw = cls.root / "fresh/raw-fp32"
        model_root = cls.refresh_raw / "old"
        model_root.mkdir(parents=True)
        cls.fresh_sources = {}
        records = {}
        for roi in reference.ROIS:
            old = cls.sources[("old", roi)]
            path = model_root / f"murtylab_old_nsd_1000_{roi}_20261011000000000Z.csv"
            matrix = old.voxels + 1.0
            frame = pd.DataFrame(matrix, columns=old.voxel_columns)
            frame.insert(0, "image", reference.canonical_image_ids())
            frame.to_csv(path, index=False)
            source = reference.load_and_validate_source(
                reference.SourceCandidate("old", roi, "20261011000000000Z", path),
                expected_ids=reference.canonical_image_ids(), expected_voxel_count=reference.VOXEL_COUNTS[roi],
                expected_means=matrix.mean(axis=1),
            )
            cls.fresh_sources[("old", roi)] = source
            records[roi] = {
                "path": path.name, "source_file": path.name, "sha256": source.sha256, "source_sha256": source.sha256,
                "voxel_columns": list(source.voxel_columns), "voxel_order_sha256": reference.sha256_json(list(source.voxel_columns)),
                "output_shape": [185, reference.VOXEL_COUNTS[roi]],
            }
        cls.proof = {
            "schema_version": 1, "model": "old", "dataset": "nsd_1000", "reference_count": 185,
            "image_ids": list(reference.canonical_image_ids()), "images": image_records,
            "image_set_sha256": reference.sha256_json(image_records),
            "subject_order": list(refresh.inference.SUBJECT_ORDER), "subject_voxel_counts": refresh.inference.SUBJECT_VOXEL_COUNTS,
            "voxel_counts": reference.VOXEL_COUNTS, "preprocessing": refresh.inference.PREPROCESSING, "seed": 0,
            "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": [],
            "numeric_policy": refresh.expansion.FP32_NUMERIC_POLICY,
            "precision_helper_sha256": cls.source_hashes["cortex/utils/inference_precision.py"],
            "core_inference_source_sha256": cls.source_hashes, "rois": records,
        }
        refresh.expansion.write_json(model_root / "provenance.json", cls.proof)
        cls.parent_sha = reference.sha256_file(cls.private / "manifest.json")
        snapshot_files = {path.relative_to(cls.baseline).as_posix(): reference.sha256_file(path)
                          for name in refresh.SNAPSHOT_ROOTS for path in (cls.baseline / name).rglob("*") if path.is_file()}
        refresh.expansion.write_json(cls.baseline / "snapshot.json", {
            "reference_version": cls.manifest["reference_version"], "models": cls.manifest["supported_models"], "files": snapshot_files,
        })
        cls.parent = copy.deepcopy(cls.manifest)
        cls.completed = cls.root / "completed-refresh"
        cls.fit_calls = []
        def fake_fit(source, **kwargs):
            cls.fit_calls.append(((source.candidate.model, source.candidate.roi), kwargs))
            return fixtures.fixture_arrays(source), {"fixture": True}
        with patch.object(refresh.expansion, "cluster_single_voxel_csv", side_effect=fixtures.fake_cluster), \
             patch.object(refresh.expansion, "cluster_single_voxel_csv_hierarchical_corr", side_effect=fixtures.fake_cluster), \
             patch.object(reference, "fit_reference_arrays", side_effect=fake_fit), \
             patch.object(refresh.inference, "core_source_identity", return_value={"core_inference_source_sha256": cls.source_hashes}):
            cls.refreshed = cls.build(cls.completed)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def build(cls, stage, **kwargs):
        return refresh.build_refreshed_reference(
            models=["old"], raw_root=cls.refresh_raw, stage_root=stage, baseline_root=cls.baseline,
            core_registry=cls.core, app_constants=cls.app, image_root=cls.images,
            parent_version=cls.parent["reference_version"], parent_manifest_sha256=cls.parent_sha,
            generated_at="2026-10-11T00:00:00Z", **kwargs,
        )

    def test_full_refresh_refits_every_existing_model_with_original_defaults(self):
        self.assertEqual(self.fit_calls, [(("old", roi), {}) for roi in sorted(reference.ROIS)])
        self.assertEqual(self.refreshed["clustering"], self.parent["clustering"])
        self.assertEqual(self.refreshed["artifact_schema"], self.parent["artifact_schema"])
        self.assertNotEqual(self.refreshed["reference_version"], self.parent["reference_version"])
        for roi in reference.ROIS:
            self.assertNotEqual(self.refreshed["artifacts"]["old"][roi]["sha256"], self.parent["artifacts"]["old"][roi]["sha256"])

    def test_all_means_are_recomputed_not_appended_from_legacy_values(self):
        old = pd.read_csv(self.data / "combined_image_means.csv")
        new = pd.read_csv(self.completed / "public/data/combined_image_means.csv")
        for roi in reference.ROIS:
            np.testing.assert_allclose(new[f"old_{roi}"] - old[f"old_{roi}"], 1.0, rtol=0, atol=1e-14)
            np.testing.assert_allclose(new[f"old_{roi}"], self.fresh_sources[("old", roi)].means, atol=1e-14)

    def test_historical_clusters_shared_visual_and_all_parent_inputs_are_preserved(self):
        snapshot = refresh.document(self.baseline / "snapshot.json")
        for relative, sha in snapshot["files"].items():
            self.assertEqual(reference.sha256_file(self.baseline / relative), sha)
            if relative.startswith("public/data/") and relative.removeprefix("public/data/") not in refresh.REWRITTEN_PUBLIC_PATHS:
                self.assertEqual(reference.sha256_file(self.completed / relative), sha)
        self.assertEqual((self.completed / "artifacts/reference/visual.npz").read_bytes(), (self.private / "visual.npz").read_bytes())
        self.assertEqual(self.refreshed["optional_artifacts"], self.parent["optional_artifacts"])
        for directory in refresh.expansion.CLUSTER_DIRS:
            summary = refresh.expansion.read_json(self.completed / f"public/data/{directory}/voxel_cluster_summary.json")
            self.assertEqual(len(summary), 3)
            self.assertTrue(all(item["source_file"].endswith("20261011000000000Z.csv") for item in summary))

    def test_full_provenance_replaces_append_only_claim_and_covers_old_models(self):
        self.assertNotIn("coverage_extension", self.refreshed)
        lineage = self.refreshed["prediction_standardization"]
        self.assertEqual(lineage["policy"], refresh.POLICY)
        self.assertEqual(lineage["analysis_input_policy"], refresh.ANALYSIS_INPUT_POLICY)
        self.assertEqual(lineage["parent_manifest_sha256"], self.parent_sha)
        self.assertEqual(lineage["parent_snapshot_sha256"], reference.sha256_file(self.baseline / "snapshot.json"))
        self.assertEqual(lineage["regenerated_models"], ["old"])
        for name in ("inference_provenance_sha256", "prediction_policies", "model_prediction_policies"):
            self.assertEqual(set(lineage[name]), {"old"})
        public = refresh.document(self.completed / "public/data/reference_manifest.json")
        self.assertEqual(public["model_prediction_policies"], lineage["model_prediction_policies"])
        self.assertNotIn("prediction_standardization", public)
        verifier.validate_refresh_lineage(self.refreshed, self.parent, self.baseline)

    def test_partial_different_or_ceiling_rosters_cannot_enter_full_refresh(self):
        for models in ([], ["new"], ["old", "new"], ["old", "old"], ["old", "ceiling"]):
            with self.subTest(models=models), self.assertRaises(reference.ReferenceValidationError):
                refresh.validate_refresh_roster(models, self.parent, self.data / "murty185_zero_index.json")

    def test_existing_staging_destination_is_not_replaced(self):
        before = reference.sha256_file(self.completed / "artifacts/reference/manifest.json")
        with self.assertRaisesRegex(reference.ReferenceValidationError, "already exists"):
            self.build(self.completed)
        self.assertEqual(reference.sha256_file(self.completed / "artifacts/reference/manifest.json"), before)

    def test_cli_full_parent_flag_passes_explicit_paths_and_complete_roster(self):
        argv = ["refresh_reference_coverage.py", "--all-parent-models", "--baseline-root", str(self.baseline),
                "--raw-root", str(self.refresh_raw), "--stage-root", "fresh-stage"]
        with patch("sys.argv", argv), patch.object(refresh, "build_refreshed_reference", return_value=self.refreshed) as build, patch("builtins.print"):
            refresh.main()
        build.assert_called_once_with(models=["old"], raw_root=self.refresh_raw,
                                      stage_root=Path("fresh-stage"), baseline_root=self.baseline, workers=1)

    def test_snapshot_requires_full_file_coverage_and_exact_parent_bytes(self):
        path = self.baseline / "snapshot.json"
        original = path.read_bytes()
        snapshot = json.loads(original)
        snapshot["files"].pop(next(iter(snapshot["files"])))
        try:
            refresh.expansion.write_json(path, snapshot)
            with self.assertRaisesRegex(reference.ReferenceValidationError, "cover all"):
                refresh.validate_parent_snapshot(self.baseline, parent_version=self.parent["reference_version"], parent_manifest_sha256=self.parent_sha)
        finally:
            path.write_bytes(original)
        with self.assertRaisesRegex(reference.ReferenceValidationError, "reviewed release"):
            refresh.validate_parent_snapshot(self.baseline, parent_version="f" * 64, parent_manifest_sha256=self.parent_sha)

    def test_current_helper_identity_core_sources_image_order_and_dimensions_are_strict(self):
        path = self.refresh_raw / "old/provenance.json"
        original = path.read_bytes()
        variants = []
        for name, value in (("seed", True), ("image_ids", list(reversed(self.proof["image_ids"]))),
                            ("precision_helper_sha256", "0" * 64),
                            ("core_inference_source_sha256", {"cortex/utils/inference_precision.py": self.proof["precision_helper_sha256"]})):
            altered = copy.deepcopy(self.proof)
            altered[name] = value
            variants.append(altered)
        altered = copy.deepcopy(self.proof)
        altered["rois"]["ffa"]["output_shape"] = [184, 716]
        variants.append(altered)
        altered = copy.deepcopy(self.proof)
        altered["numeric_policy"]["cuda_matmul_allow_tf32"] = True
        variants.append(altered)
        try:
            for altered in variants:
                refresh.expansion.write_json(path, altered)
                with self.subTest(altered=altered.keys()), \
                     patch.object(refresh.inference, "core_source_identity", return_value={"core_inference_source_sha256": self.source_hashes}), \
                     self.assertRaises(reference.ReferenceValidationError):
                    refresh.validate_inference_proofs(models=["old"], raw_root=self.refresh_raw,
                                                      sources=self.fresh_sources, core_root=self.core_root, image_root=self.images)
        finally:
            path.write_bytes(original)

    def test_refreshed_lineage_rejects_hidden_partial_policy_or_false_preservation(self):
        for name in ("inference_provenance_sha256", "prediction_policies", "model_prediction_policies"):
            altered = copy.deepcopy(self.refreshed)
            altered["prediction_standardization"][name] = {}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "incomplete model lineage"):
                verifier.validate_refresh_lineage(altered, self.parent, self.baseline)
        altered = copy.deepcopy(self.refreshed)
        altered["coverage_extension"] = {"policy": "append_only_preserve_baseline_artifacts_v1"}
        with self.assertRaisesRegex(ValueError, "unchanged scientific method"):
            verifier.validate_refresh_lineage(altered, self.parent, self.baseline)
        for policy in (None, "decimal_csv_float64", ""):
            altered = copy.deepcopy(self.refreshed)
            if policy is None:
                altered["prediction_standardization"].pop("analysis_input_policy")
            else:
                altered["prediction_standardization"]["analysis_input_policy"] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "unchanged scientific method"):
                verifier.validate_refresh_lineage(altered, self.parent, self.baseline)

    def test_refresh_rejects_changed_cluster_labels_or_non_sign_plot_alignment(self):
        for failure in ("labels", "axes"):
            stage = self.root / f"failed-refresh-{failure}"
            def altered_fit(source):
                arrays = fixtures.fixture_arrays(source)
                if failure == "labels":
                    arrays["hierarchical_labels"][0] = 7
                else:
                    arrays["plot_reference_coordinates"] *= 2.0
                return arrays, {}
            with self.subTest(failure=failure), \
                 patch.object(refresh.expansion, "cluster_single_voxel_csv", side_effect=fixtures.fake_cluster), \
                 patch.object(refresh.expansion, "cluster_single_voxel_csv_hierarchical_corr", side_effect=fixtures.fake_cluster), \
                 patch.object(reference, "fit_reference_arrays", side_effect=altered_fit), \
                 patch.object(refresh.inference, "core_source_identity", return_value={"core_inference_source_sha256": self.source_hashes}), \
                 self.assertRaises(reference.ReferenceValidationError):
                self.build(stage)
            self.assertFalse(stage.exists())
            self.assertEqual(list(self.root.glob(f".{stage.name}-refresh-*")), [])

    def test_refreshed_source_cannot_reuse_a_historical_filename(self):
        from dataclasses import replace
        source = self.fresh_sources[("old", "ffa")]
        collision = replace(source, candidate=replace(source.candidate,
                            path=self.baseline / "data/image_by_model" / self.parent["artifacts"]["old"]["ffa"]["source_file"]))
        sources = {**self.fresh_sources, ("old", "ffa"): collision}
        stage = self.root / "rejected-collision-stage"
        with patch.object(refresh.expansion, "load_new_sources", return_value=(sources, {})), \
             patch.object(refresh.expansion, "cluster_single_voxel_csv") as clustering, \
             self.assertRaisesRegex(reference.ReferenceValidationError, "collides with a retained historical"):
            self.build(stage)
        clustering.assert_not_called()
        self.assertFalse(stage.exists())

    def test_verifier_rejects_an_incomplete_shared_preservation_report_before_core_load(self):
        report_path = self.completed / "preservation_report.json"
        original_document = verifier.document
        bad_report = original_document(report_path)
        bad_report["public_artifact_sha256"].pop(next(iter(bad_report["public_artifact_sha256"])))
        def documents(path):
            return bad_report if path == report_path else original_document(path)
        with patch.object(verifier, "document", side_effect=documents), self.assertRaisesRegex(ValueError, "omitted or changed shared files"):
            verifier.verify_refresh_release(self.completed, self.root / "unused-spatial", repo=self.root,
                                            baseline=self.baseline, raw_root=self.refresh_raw,
                                            parent_version=self.parent["reference_version"], parent_manifest_sha256=self.parent_sha)


class CanonicalFloat32AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.csv = cls.root / "murtylab_near_flat_nsd_1000_ffa_20261011000000000Z.csv"
        rng = np.random.default_rng(42)
        cls.bits = (0.1 + rng.normal(size=(185, 12)) * 3e-8).astype(np.float32)
        frame = pd.DataFrame(cls.bits, columns=[f"v{i}" for i in range(12)])
        frame.insert(0, "image", reference.canonical_image_ids())
        frame.to_csv(cls.csv, index=False, float_format="%.9g")
        decimal = pd.read_csv(cls.csv).drop(columns="image").to_numpy(dtype=np.float64)
        cls.source = reference.load_and_validate_source(
            reference.SourceCandidate("near_flat", "ffa", "20261011000000000Z", cls.csv),
            expected_ids=reference.canonical_image_ids(), expected_voxel_count=12,
            expected_means=decimal.mean(axis=1),
        )
        cls.canonical = refresh.canonicalize_source(cls.source)
        cls.arrays, cls.metadata = reference.fit_reference_arrays(cls.canonical)
        cls.serial = cls.root / "serial"
        cls.parallel = cls.root / "parallel"
        for root in (cls.serial, cls.parallel):
            for directory in refresh.expansion.CLUSTER_DIRS:
                (root / directory).mkdir(parents=True)
        cls.serial_summaries = refresh.cluster_refreshed_source((cls.csv, cls.serial))
        # Four independently named inputs exercise four real worker processes,
        # rather than claiming equivalence from a mocked clustering result.
        cls.parallel_jobs = []
        for i in range(4):
            path = cls.csv if i == 0 else cls.root / f"murtylab_near_flat_{i}_nsd_1000_ffa_20261011000000000Z.csv"
            if i:
                shutil.copyfile(cls.csv, path)
            cls.parallel_jobs.append((path, cls.parallel))
        with ProcessPoolExecutor(max_workers=4) as executor:
            cls.parallel_summaries = list(executor.map(refresh.cluster_refreshed_source, cls.parallel_jobs))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_near_flat_means_and_scaler_use_exact_saved_float32_bits(self):
        np.testing.assert_array_equal(self.canonical.voxels, self.bits.astype(np.float64))
        self.assertEqual(self.canonical.sha256, self.source.sha256)
        self.assertIs(self.canonical.candidate, self.source.candidate)
        np.testing.assert_array_equal(self.canonical.means, self.bits.mean(axis=1, dtype=np.float64))
        np.testing.assert_array_equal(self.arrays["reference_voxels"], self.bits)
        np.testing.assert_array_equal(self.arrays["reference_means"], self.bits.mean(axis=1, dtype=np.float64))
        transformed = (self.bits.astype(np.float64) - self.arrays["scaler_mean"]) / self.arrays["scaler_scale"]
        np.testing.assert_allclose(self.arrays["standardized_reference_matrix"], transformed, rtol=1e-6, atol=1e-6)
        # Demonstrate that the fixture genuinely catches the previous bug.
        decimal_standardized = (self.source.voxels - self.arrays["scaler_mean"]) / self.arrays["scaler_scale"]
        self.assertGreater(np.max(np.abs(decimal_standardized - transformed)), 1e-4)

    def test_near_flat_artifact_passes_actual_unchanged_core_validation(self):
        core = reference.REPO_ROOT.parent / "cortex-core"
        if not (core / "cortex/comparison/artifacts.py").is_file():
            self.skipTest("Neighboring Cortex Core checkout is required for this integration check")
        sys.path.insert(0, str(core))
        from cortex.comparison.artifacts import ArtifactRepository, ReferenceArtifact
        kwargs = {name: self.arrays[name] for name in ReferenceArtifact.__dataclass_fields__ if name in self.arrays}
        artifact = ReferenceArtifact(reference_version="a" * 64, model="near_flat", roi="ffa", **kwargs)
        ArtifactRepository._validate_artifact(artifact, expected_count=185, expected_voxels=12)

    def test_verifier_compares_canonical_means_not_decimal_csv_means(self):
        with self.assertRaisesRegex(reference.ReferenceValidationError, "row means"):
            reference.load_and_validate_source(
                self.source.candidate, expected_ids=reference.canonical_image_ids(),
                expected_voxel_count=12, expected_means=self.canonical.means,
            )
        verified = verifier.load_canonical_source(
            self.source.candidate, expected_voxel_count=12, expected_means=self.canonical.means,
        )
        np.testing.assert_array_equal(verified.voxels, self.canonical.voxels)
        np.testing.assert_array_equal(verified.means, self.canonical.means)
        wrong_means = self.canonical.means.copy()
        wrong_means[0] += 1e-7
        with self.assertRaisesRegex(ValueError, "canonical row means"):
            verifier.load_canonical_source(
                self.source.candidate, expected_voxel_count=12, expected_means=wrong_means,
            )

    def test_static_labels_and_projection_match_private_canonical_fit(self):
        contracts = reference.load_static_voxel_cluster_contracts(
            {("near_flat", "ffa"): self.canonical},
            kmeans_dir=self.serial / refresh.expansion.CLUSTER_DIRS[0],
            hierarchical_dir=self.serial / refresh.expansion.CLUSTER_DIRS[1],
        )
        contract = contracts[("near_flat", "ffa")]
        np.testing.assert_array_equal(self.arrays["kmeans_labels"], contract.kmeans_labels)
        np.testing.assert_array_equal(self.arrays["hierarchical_labels"], contract.hierarchical_labels)
        _, alignment = reference.align_plot_projection_to_static(
            self.arrays, contract.plot_coordinates, context="near_flat/ffa", residual_tolerance=1e-9,
        )
        self.assertLess(alignment["plot_alignment_max_abs_error"], 1e-9)

    def test_four_workers_produce_identical_static_bytes_and_normalized_summaries(self):
        for directory, serial_summary, parallel_summary in zip(
            refresh.expansion.CLUSTER_DIRS, self.serial_summaries, self.parallel_summaries[0],
        ):
            old = {p.name: p.read_bytes() for p in (self.serial / directory).iterdir() if not p.name.endswith("_summary.json")}
            new = {p.name: p.read_bytes() for p in (self.parallel / directory).iterdir() if p.name.startswith(self.csv.stem) and not p.name.endswith("_summary.json")}
            self.assertEqual(old, new)
            self.assertEqual(refresh.expansion._normalized_summary(serial_summary, directory),
                             refresh.expansion._normalized_summary(parallel_summary, directory))

    def test_worker_count_is_explicit_and_bounded(self):
        for workers in (0, 5, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(reference.ReferenceValidationError, "one to four"):
                refresh.build_refreshed_reference(models=[], raw_root=self.root, stage_root=self.root / "unused", workers=workers)

    def test_legacy_cluster_ingestion_default_remains_decimal_float64(self):
        for function in (refresh.expansion.cluster_single_voxel_csv, refresh.expansion.cluster_single_voxel_csv_hierarchical_corr):
            import inspect
            self.assertIs(inspect.signature(function).parameters["canonical_float32"].default, False)


if __name__ == "__main__":
    unittest.main()
