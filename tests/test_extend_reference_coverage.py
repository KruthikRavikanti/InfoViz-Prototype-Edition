from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.data import build_reference_artifacts as reference
from scripts.data import extend_reference_coverage as expansion
from scripts.data.validate_model_registries import RegistryValidationError, validate_model_identifiers


def fixture_arrays(source: reference.ValidatedSource) -> dict[str, np.ndarray]:
    """Small synthetic transform contract; fitting is mocked, never real inference."""
    arrays = {name: np.zeros(1, dtype=np.float64) for name in reference.ARRAY_CONTRACT}
    arrays.update({
        "image_ids": np.asarray(source.image_ids),
        "reference_means": source.means,
        "reference_voxels": source.voxels.astype(np.float32),
        "kmeans_labels": (np.arange(185) % 2).astype(np.int32),
        "hierarchical_labels": (np.arange(185) % 2).astype(np.int32),
        "hierarchical_medoid_ids": np.asarray([source.image_ids[0], source.image_ids[1]]),
        "plot_reference_coordinates": np.column_stack((np.arange(185), -np.arange(185))).astype(np.float64),
        "plot_pca_components": np.zeros((2, source.voxels.shape[1])),
    })
    return arrays


def fake_cluster(source_path: Path, output: Path, **kwargs) -> dict:
    frame = pd.read_csv(source_path)
    match = reference.FILENAME_RE.fullmatch(source_path.name)
    assert match is not None
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"{source_path.stem}_clusters.csv"
    pd.DataFrame({
        "image_name": reference.canonical_image_ids(),
        "cluster_label": np.arange(185) % 2,
        "pca_x": np.arange(185, dtype=np.float64),
        "pca_y": -np.arange(185, dtype=np.float64),
    }).to_csv(result, index=False)
    return {
        "source_file": source_path.name, "result_csv": str(result),
        "model": match.group("model"), "roi": match.group("roi"),
        "n_images": 185, "n_voxels": frame.shape[1] - 1, "best_k": 2,
        "silhouette": 0.5, "cluster_sizes": {"0": 93, "1": 92},
    }


class ExpansionHelpersTests(unittest.TestCase):
    def test_target_must_preserve_baseline_and_add_unique_safe_models(self):
        self.assertEqual(expansion.validate_target_roster(["old", "new"], ["old"]), ("new", "old"))
        for models in (["old"], ["new"], ["old", "new", "new"], ["old", "../new"]):
            with self.subTest(models=models), self.assertRaises(reference.ReferenceValidationError):
                expansion.validate_target_roster(models, ["old"])

    def test_registry_matching_does_not_guess_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = root / "registry.py"
            app = root / "constants.jsx"
            core.write_text("backbone_models = {'one': {'name': 'clip_rn50'}}\n")
            app.write_text('export const MODEL_OPTIONS = [{"value": "clip_rn50"}];\n')
            self.assertEqual(validate_model_identifiers(["clip_rn50"], core_registry=core, app_constants=app), ("clip_rn50",))
            for models in (["CLIP_RN50"], ["clip_rn_50"], ["clip_rn50", "clip_rn50"]):
                with self.subTest(models=models), self.assertRaises(RegistryValidationError):
                    validate_model_identifiers(models, core_registry=core, app_constants=app)

    def test_verified_decimal_model_identifiers_are_safe_but_traversal_is_not(self):
        model = "toponets_rn18_tau_0.5"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = root / "registry.py"
            app = root / "constants.jsx"
            core.write_text(f"backbone_models = {{'one': {{'name': '{model}'}}}}\n")
            app.write_text(f'export const MODEL_OPTIONS = [{{"value":"{model}"}}];\n')
            self.assertEqual(validate_model_identifiers([model], core_registry=core, app_constants=app), (model,))
            for invalid in ("../0.5", "toponets_rn18_tau_0/5", ".5", ".."):
                with self.subTest(invalid=invalid), self.assertRaises(RegistryValidationError):
                    validate_model_identifiers([invalid], core_registry=core, app_constants=app)

    def test_explicit_source_discovery_keeps_legacy_default_count_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for roi in reference.ROIS:
                (root / f"murtylab_new_nsd_1000_{roi}_20261010000000000Z.csv").write_text("image,v1\nimage_001.png,1\n")
            self.assertEqual(len(reference.discover_source_candidates(root, expected_models=["new"])), 3)
            with self.assertRaisesRegex(reference.ReferenceValidationError, "exactly 26"):
                reference.discover_source_candidates(root)
            with self.assertRaisesRegex(reference.ReferenceValidationError, "explicit model roster"):
                reference.discover_source_candidates(root, expected_models=["different"])

    def test_summary_merge_rejects_duplicates_missing_and_wrong_dimensions(self):
        old = [{"model": "old", "roi": roi, "n_images": 185, "n_voxels": reference.VOXEL_COUNTS[roi]}
               for roi in reference.ROIS]
        new = [{**row, "model": "new"} for row in old]
        self.assertEqual(expansion.merge_cluster_summaries(old, new, ["old", "new"])[:3], old)
        for additions in (new[:-1], new + [new[0]], [{**row, "n_images": 184} for row in new]):
            with self.subTest(additions=additions), self.assertRaises(reference.ReferenceValidationError):
                expansion.merge_cluster_summaries(old, additions, ["old", "new"])

    def test_preservation_validator_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            old.mkdir(); new.mkdir()
            (old / "model.npz").write_bytes(b"baseline")
            (new / "model.npz").write_bytes(b"baseline")
            self.assertEqual(expansion.assert_preserved_files(old, new)["model.npz"], reference.sha256_file(old / "model.npz"))
            (new / "model.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(reference.ReferenceValidationError, "changed baseline artifact bytes"):
                expansion.assert_preserved_files(old, new)

    def test_ceiling_cannot_be_requested_as_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            expansion.write_json(path, {roi: {"old": 0.5, "new": 0.6, "ceiling": 0.8} for roi in reference.ROIS})
            expansion.validate_scoreboard_roster(["old", "new"], path)
            for models in (["ceiling"], ["unscored"]):
                with self.assertRaisesRegex(reference.ReferenceValidationError, "non-model/unscored"):
                    expansion.validate_scoreboard_roster(models, path)

    def test_stochastic_vone_policy_cannot_hide_neuronal_noise_or_claim_determinism(self):
        deterministic = {"response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": []}
        stochastic = {"response_policy": "seeded_stochastic_sampling", "deterministic_per_stimulus": False, "noise_modes": ["neuronal"]}
        self.assertEqual(expansion.prediction_policy_from_provenance("resnet50", deterministic), deterministic)
        for model in expansion.STOCHASTIC_VONE_MODELS:
            self.assertEqual(expansion.prediction_policy_from_provenance(model, stochastic), stochastic)
            for altered in (deterministic, {**stochastic, "deterministic_per_stimulus": True},
                            {**stochastic, "noise_modes": []}, {**stochastic, "response_policy": "averaged"}):
                with self.subTest(model=model, altered=altered), self.assertRaises(reference.ReferenceValidationError):
                    expansion.prediction_policy_from_provenance(model, altered)
        with self.assertRaises(reference.ReferenceValidationError):
            expansion.prediction_policy_from_provenance("resnet50", stochastic)

    def test_vone_rn50_registered_no_noise_constructor_is_deterministic(self):
        deterministic = {"response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": []}
        self.assertNotIn("vone_rn50", expansion.STOCHASTIC_VONE_MODELS)
        self.assertEqual(expansion.prediction_policy_from_provenance("vone_rn50", deterministic), deterministic)
        with self.assertRaises(reference.ReferenceValidationError):
            expansion.prediction_policy_from_provenance("vone_rn50", {
                "response_policy": "seeded_stochastic_sampling",
                "deterministic_per_stimulus": False, "noise_modes": ["neuronal"],
            })

    def test_precision_policy_records_observed_fp32_flags_and_helper_identity(self):
        expected = {"numeric_policy": dict(expansion.FP32_NUMERIC_POLICY), "precision_helper_sha256": "a" * 64}
        provenance = {**expected, "core_inference_source_sha256": {"cortex/utils/inference_precision.py": "a" * 64}}
        checked = expansion.precision_policy_from_provenance("new", provenance)
        self.assertEqual(checked, expected)
        self.assertIsNot(checked["numeric_policy"], provenance["numeric_policy"])

    def test_precision_policy_rejects_tf32_amp_missing_flags_and_false_as_integer(self):
        correct = dict(expansion.FP32_NUMERIC_POLICY)
        altered_policies = [None, {**correct, "policy": "tf32"}, {**correct, "output_dtype": "float16"},
                            {**correct, "float32_matmul_precision": "high"}, {**correct, "extra": False},
                            {key: value for key, value in correct.items() if key != "cudnn_allow_tf32"}]
        for key, value in correct.items():
            if isinstance(value, bool):
                altered_policies.extend(({**correct, key: not value}, {**correct, key: int(value)}))
        for numeric in altered_policies:
            with self.subTest(numeric=numeric), self.assertRaises(reference.ReferenceValidationError):
                expansion.precision_policy_from_provenance("new", {
                    "numeric_policy": numeric, "precision_helper_sha256": "a" * 64,
                })

    def test_precision_policy_requires_real_source_hash(self):
        for digest in (None, "", "a" * 63, "A" * 64, "../helper.py", 1):
            with self.subTest(digest=digest), self.assertRaises(reference.ReferenceValidationError):
                expansion.precision_policy_from_provenance("new", {
                    "numeric_policy": dict(expansion.FP32_NUMERIC_POLICY), "precision_helper_sha256": digest,
                })

    def test_precision_helper_identity_must_agree_with_core_source_lineage(self):
        for source_hashes in (None, {}, {"cortex/utils/inference_precision.py": "b" * 64}):
            with self.subTest(source_hashes=source_hashes), self.assertRaises(reference.ReferenceValidationError):
                expansion.precision_policy_from_provenance("new", {
                    "numeric_policy": dict(expansion.FP32_NUMERIC_POLICY), "precision_helper_sha256": "a" * 64,
                    "core_inference_source_sha256": source_hashes,
                })


class ExpansionFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.private = cls.root / "baseline/artifacts/reference"
        cls.data = cls.root / "baseline/public/data"
        cls.raw = cls.root / "baseline/raw"
        cls.new_raw = cls.root / "new/raw"
        for directory in (cls.private, cls.data, cls.raw, cls.new_raw / "new"):
            directory.mkdir(parents=True)
        cls.core = cls.root / "registry.py"
        cls.app = cls.root / "constants.jsx"
        cls.core.write_text("backbone_models = {'old': {'name': 'old'}, 'new': {'name': 'new'}}\n")
        cls.app.write_text('export const MODEL_OPTIONS = [{"value":"old"},{"value":"new"}];\n')
        sources = {}
        for model, raw_directory, offset in (("old", cls.raw, 0), ("new", cls.new_raw / "new", 10)):
            for roi in reference.ROIS:
                path = raw_directory / f"murtylab_{model}_nsd_1000_{roi}_20261010000000000Z.csv"
                columns = [f"v{index}" for index in range(reference.VOXEL_COUNTS[roi])]
                values = (np.arange(185)[:, None] + np.arange(len(columns))[None, :] + offset) / 1024
                frame = pd.DataFrame(values, columns=columns)
                frame.insert(0, "image", reference.canonical_image_ids())
                frame.to_csv(path, index=False)
                candidate = reference.SourceCandidate(model, roi, "20261010000000000Z", path)
                sources[(model, roi)] = reference.load_and_validate_source(
                    candidate, expected_ids=reference.canonical_image_ids(),
                    expected_voxel_count=len(columns), expected_means=values.mean(axis=1),
                )
        cls.sources = sources
        means = pd.DataFrame({"image_name": reference.canonical_image_ids(),
                              **{f"old_{roi}": sources[("old", roi)].means for roi in reference.ROIS}})
        means.to_csv(cls.data / "combined_image_means.csv", index=False)
        expansion.write_json(cls.data / "murty185_zero_index.json", {
            roi: {"old": 0.5, "new": 0.6, "ceiling": 0.9} for roi in reference.ROIS
        })
        entries = {"old": {}}
        selection = {}
        for roi in reference.ROIS:
            source = sources[("old", roi)]
            summaries = []
            for directory in expansion.CLUSTER_DIRS:
                summary = fake_cluster(source.candidate.path, cls.data / directory)
                summaries.append(summary)
            arrays = fixture_arrays(source)
            relative = Path("models/old") / f"{roi}.npz"
            reference.write_deterministic_npz(cls.private / relative, arrays)
            contract = reference.load_static_voxel_cluster_contracts(
                {("old", roi): source}, kmeans_dir=cls.data / expansion.CLUSTER_DIRS[0],
                hierarchical_dir=cls.data / expansion.CLUSTER_DIRS[1],
            )[("old", roi)]
            entries["old"][roi] = expansion._new_artifact_entry(source, arrays, {}, contract, cls.private / relative, relative)
            selection[f"old/{roi}"] = {"selected": source.candidate.path.name}
        for directory in expansion.CLUSTER_DIRS:
            summaries = [fake_cluster(sources[("old", roi)].candidate.path, cls.data / directory) for roi in reference.ROIS]
            expansion.write_json(cls.data / directory / "voxel_cluster_summary.json", summaries)
            pd.DataFrame(summaries).to_csv(cls.data / directory / "voxel_cluster_summary.csv", index=False)
        (cls.private / "visual.npz").write_bytes(b"fixture immutable visual artifact")
        cls.manifest = {
            "schema_version": 1, "reference_version": "a" * 64, "checksum": "a" * 64,
            "dataset": "nsd_1000", "reference_count": 185, "image_count": 185,
            "rois": list(reference.ROIS), "supported_models": ["old"],
            "voxel_counts": reference.VOXEL_COUNTS,
            "image_ids_sha256": reference.sha256_json(list(reference.canonical_image_ids())),
            "clustering": reference._clustering_manifest(), "artifact_schema": reference._artifact_schema_manifest(),
            "source_selection": selection, "artifacts": entries,
            "combined_means": {"file": "combined_image_means.csv", "sha256": reference.sha256_file(cls.data / "combined_image_means.csv")},
            "optional_artifacts": {"visual": {
                "available": True, "path": "visual.npz", "sha256": reference.sha256_file(cls.private / "visual.npz"),
                "feature_schema": "infoviz_visual_77_v1", "source_sha256": "b" * 64, "source_images_sha256": "c" * 64,
                "static_kmeans_sha256": "d" * 64, "static_hierarchical_sha256": "e" * 64,
            }, "dreamsim": {"available": False}},
            "library_versions": reference._library_versions(), "generated_at": "2026-10-10T00:00:00Z",
            "generator_git": {"commit": "fixture", "dirty": False},
        }
        expansion.write_json(cls.private / "manifest.json", cls.manifest)
        expansion.write_json(cls.data / "reference_manifest.json", reference.public_manifest_from_full(cls.manifest))
        expansion.write_json(cls.new_raw / "new/provenance.json", {
            "schema_version": 1, "model": "new", "dataset": "nsd_1000",
            "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": [],
            "numeric_policy": dict(expansion.FP32_NUMERIC_POLICY), "precision_helper_sha256": "b" * 64,
            "core_inference_source_sha256": {"cortex/utils/inference_precision.py": "b" * 64},
            "rois": {roi: {"source_file": sources[("new", roi)].candidate.path.name,
                           "source_sha256": sources[("new", roi)].sha256} for roi in reference.ROIS},
        })

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_stage_fits_only_added_models_preserving_all_existing_artifacts(self):
        stage = self.root / "completed-stage"
        fitted = []
        def fake_fit(source):
            fitted.append((source.candidate.model, source.candidate.roi))
            return fixture_arrays(source), {"fixture": True}
        private_before = {path.relative_to(self.private): reference.sha256_file(path) for path in self.private.rglob("*") if path.is_file()}
        data_before = {path.relative_to(self.data): reference.sha256_file(path) for path in self.data.rglob("*") if path.is_file()}
        with patch.object(expansion, "cluster_single_voxel_csv", side_effect=fake_cluster), \
             patch.object(expansion, "cluster_single_voxel_csv_hierarchical_corr", side_effect=fake_cluster), \
             patch.object(reference, "fit_reference_arrays", side_effect=fake_fit):
            manifest = expansion.build_expanded_reference(
                models=["old", "new"], new_raw_root=self.new_raw, stage_root=stage,
                baseline_artifact_root=self.private, baseline_data_root=self.data, baseline_raw_root=self.raw,
                core_registry=self.core, app_constants=self.app, generated_at="2026-10-10T00:00:00Z",
            )
        self.assertEqual(set(fitted), {("new", roi) for roi in reference.ROIS})
        self.assertEqual(manifest["artifacts"]["old"], self.manifest["artifacts"]["old"])
        self.assertEqual(manifest["coverage_extension"]["parent_reference_version"], "a" * 64)
        self.assertNotEqual(manifest["reference_version"], "a" * 64)
        self.assertEqual(manifest["supported_models"], ["new", "old"])
        self.assertTrue(manifest["coverage_extension"]["model_prediction_policies"]["new"]["deterministic_per_stimulus"])
        self.assertEqual(manifest["coverage_extension"]["prediction_policies"]["new"], {
            "numeric_policy": expansion.FP32_NUMERIC_POLICY, "precision_helper_sha256": "b" * 64,
        })
        self.assertNotIn("old", manifest["coverage_extension"]["prediction_policies"])
        public = expansion.read_json(stage / "public/data/reference_manifest.json")
        self.assertEqual(public["model_prediction_policies"], manifest["coverage_extension"]["model_prediction_policies"])
        self.assertNotIn("model_prediction_policies", reference.public_manifest_from_full(self.manifest))
        for relative, digest in private_before.items():
            self.assertEqual(reference.sha256_file(self.private / relative), digest)
            if str(relative) != "manifest.json":
                self.assertEqual(reference.sha256_file(stage / "artifacts/reference" / relative), digest)
        for relative, digest in data_before.items():
            self.assertEqual(reference.sha256_file(self.data / relative), digest)
        with (self.data / "combined_image_means.csv").open(newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        with (stage / "public/data/combined_image_means.csv").open(newline="") as handle:
            new_rows = list(csv.DictReader(handle))
        self.assertEqual([{key: row[key] for key in old_rows[0]} for row in new_rows], old_rows)
        report = expansion.read_json(stage / "preservation_report.json")
        self.assertFalse(report["live_install_performed"])

    def test_new_provenance_hash_and_voxel_order_cannot_be_guessed(self):
        baseline_sources = {(model, roi): source for (model, roi), source in self.sources.items() if model == "old"}
        checked, _ = expansion.load_new_sources(self.new_raw, ["new"], baseline_sources)
        self.assertEqual(len(checked), 3)
        path = self.new_raw / "new/provenance.json"
        original = path.read_text()
        try:
            provenance = json.loads(original)
            provenance["rois"]["ffa"]["source_sha256"] = "0" * 64
            expansion.write_json(path, provenance)
            with self.assertRaisesRegex(reference.ReferenceValidationError, "does not verify"):
                expansion.load_new_sources(self.new_raw, ["new"], baseline_sources)
        finally:
            path.write_text(original)
        altered = dict(baseline_sources)
        source = altered[("old", "ffa")]
        from dataclasses import replace
        altered[("old", "ffa")] = replace(source, voxel_columns=tuple(reversed(source.voxel_columns)))
        with self.assertRaisesRegex(reference.ReferenceValidationError, "voxel identity/order"):
            expansion.load_new_sources(self.new_raw, ["new"], altered)

    def test_legacy_or_tf32_new_provenance_is_rejected_before_any_clustering(self):
        provenance_path = self.new_raw / "new/provenance.json"
        original_read_json = expansion.read_json
        original = original_read_json(provenance_path)
        variants = [
            {key: value for key, value in original.items() if key != "numeric_policy"},
            {**original, "numeric_policy": {**original["numeric_policy"], "cuda_matmul_allow_tf32": True}},
        ]
        for index, provenance in enumerate(variants):
            stage = self.root / f"rejected-precision-stage-{index}"
            def read_with_altered_provenance(path):
                return provenance if path == provenance_path else original_read_json(path)
            with self.subTest(index=index), patch.object(expansion, "read_json", side_effect=read_with_altered_provenance), \
                 patch.object(expansion, "cluster_single_voxel_csv") as cluster:
                with self.assertRaisesRegex(reference.ReferenceValidationError, "FP32 numeric policy"):
                    expansion.build_expanded_reference(
                        models=["old", "new"], new_raw_root=self.new_raw, stage_root=stage,
                        baseline_artifact_root=self.private, baseline_data_root=self.data, baseline_raw_root=self.raw,
                        core_registry=self.core, app_constants=self.app,
                    )
                cluster.assert_not_called()
                self.assertFalse(stage.exists())
                self.assertEqual(list(self.root.glob(f".{stage.name}-build-*")), [])

    def test_pilot_path_hash_aliases_are_verified_and_disagreement_fails(self):
        path = self.new_raw / "new/provenance.json"
        original = path.read_text()
        baseline_sources = {(model, roi): source for (model, roi), source in self.sources.items() if model == "old"}
        try:
            provenance = json.loads(original)
            for metadata in provenance["rois"].values():
                metadata["path"] = metadata.pop("source_file")
                metadata["sha256"] = metadata.pop("source_sha256")
            expansion.write_json(path, provenance)
            self.assertEqual(len(expansion.load_new_sources(self.new_raw, ["new"], baseline_sources)[0]), 3)
            provenance["rois"]["ffa"]["source_sha256"] = "0" * 64
            expansion.write_json(path, provenance)
            with self.assertRaisesRegex(reference.ReferenceValidationError, "does not verify"):
                expansion.load_new_sources(self.new_raw, ["new"], baseline_sources)
        finally:
            path.write_text(original)

    def test_existing_destination_is_never_replaced(self):
        stage = self.root / "occupied-stage"
        stage.mkdir()
        (stage / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(reference.ReferenceValidationError, "already exists"):
            expansion.build_expanded_reference(models=["old", "new"], new_raw_root=self.new_raw, stage_root=stage)
        self.assertEqual((stage / "keep.txt").read_text(), "keep")

    def test_failed_generation_leaves_no_stage_and_never_mutates_baseline(self):
        stage = self.root / "failed-stage"
        old_digest = reference.sha256_file(self.private / "manifest.json")
        with patch.object(expansion, "cluster_single_voxel_csv", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                expansion.build_expanded_reference(
                    models=["old", "new"], new_raw_root=self.new_raw, stage_root=stage,
                    baseline_artifact_root=self.private, baseline_data_root=self.data, baseline_raw_root=self.raw,
                    core_registry=self.core, app_constants=self.app,
                )
        self.assertFalse(stage.exists())
        self.assertEqual(list(self.root.glob(".failed-stage-build-*")), [])
        self.assertEqual(reference.sha256_file(self.private / "manifest.json"), old_digest)

    def test_fitted_labels_must_match_static_clusters_before_stage_publication(self):
        stage = self.root / "mismatched-label-stage"
        def wrong_fit(source):
            arrays = fixture_arrays(source)
            arrays["kmeans_labels"] = 1 - arrays["kmeans_labels"]
            return arrays, {}
        with patch.object(expansion, "cluster_single_voxel_csv", side_effect=fake_cluster), \
             patch.object(expansion, "cluster_single_voxel_csv_hierarchical_corr", side_effect=fake_cluster), \
             patch.object(reference, "fit_reference_arrays", side_effect=wrong_fit):
            with self.assertRaisesRegex(reference.ReferenceValidationError, "cluster labels disagree"):
                expansion.build_expanded_reference(
                    models=["old", "new"], new_raw_root=self.new_raw, stage_root=stage,
                    baseline_artifact_root=self.private, baseline_data_root=self.data, baseline_raw_root=self.raw,
                    core_registry=self.core, app_constants=self.app,
                )
        self.assertFalse(stage.exists())


if __name__ == "__main__":
    unittest.main()
