from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.build_reference_artifacts import (
    ARRAY_CONTRACT,
    REPO_ROOT,
    SourceCandidate,
    ValidatedSource,
    align_plot_projection_to_static,
    canonical_image_ids,
    discover_source_candidates,
    extract_and_validate_visual_features,
    fit_reference_arrays,
    fit_visual_reference_arrays,
    load_and_validate_source,
    load_combined_means,
    sha256_file,
    write_deterministic_npz,
)
from scripts.data.validate_model_registries import validate_registries


class ReferenceArtifactUnitTests(unittest.TestCase):
    @staticmethod
    def _candidate(path: Path) -> SourceCandidate:
        return SourceCandidate(
            model="fixture",
            roi="ffa",
            timestamp="20260101000000000Z",
            path=path,
        )

    def test_malformed_fixture_rejects_non_numeric_voxel(self) -> None:
        path = REPO_ROOT / "tests" / "fixtures" / "reference_artifacts" / "non_numeric.csv"
        candidate = SourceCandidate(
            model="fixture",
            roi="ffa",
            timestamp="20260101000000000Z",
            path=path,
        )
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            load_and_validate_source(
                candidate,
                expected_ids=("image_001.png", "image_002.png", "image_003.png"),
                expected_voxel_count=2,
                expected_means=np.asarray([1.5, 3.5, 5.5], dtype=np.float64),
            )

    def test_duplicate_image_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            pd.DataFrame(
                {
                    "image": ["image_001.png", "image_001.png", "image_003.png"],
                    "v1": [1.0, 3.0, 5.0],
                    "v2": [2.0, 4.0, 6.0],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate image identifiers"):
                load_and_validate_source(
                    self._candidate(path),
                    expected_ids=("image_001.png", "image_002.png", "image_003.png"),
                    expected_voxel_count=2,
                    expected_means=np.asarray([1.5, 3.5, 5.5]),
                )

    def test_nan_and_infinity_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "non_finite.csv"
                pd.DataFrame(
                    {
                        "image": ["image_001.png", "image_002.png", "image_003.png"],
                        "v1": [1.0, value, 5.0],
                        "v2": [2.0, 4.0, 6.0],
                    }
                ).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    load_and_validate_source(
                        self._candidate(path),
                        expected_ids=("image_001.png", "image_002.png", "image_003.png"),
                        expected_voxel_count=2,
                        expected_means=np.asarray([1.5, 3.5, 5.5]),
                    )

    def test_wrong_voxel_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_width.csv"
            pd.DataFrame(
                {
                    "image": ["image_001.png", "image_002.png", "image_003.png"],
                    "v1": [1.0, 3.0, 5.0],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "expected 2 voxel columns"):
                load_and_validate_source(
                    self._candidate(path),
                    expected_ids=("image_001.png", "image_002.png", "image_003.png"),
                    expected_voxel_count=2,
                    expected_means=np.asarray([1.5, 3.5, 5.5]),
                )

    def test_incomplete_model_roi_triad_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            for roi in ("ffa", "ppa"):
                (raw_dir / f"murtylab_fixture_nsd_1000_{roi}_20260101000000000Z.csv").write_text(
                    "image,v1\nimage_001.png,1\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "complete FFA/PPA/EBA triad"):
                discover_source_candidates(raw_dir, expected_model_count=1)

    def test_deterministic_npz_has_stable_hash_and_no_pickle(self) -> None:
        arrays = {
            name: np.asarray([index], dtype=np.float64)
            for index, name in enumerate(sorted(ARRAY_CONTRACT))
        }
        arrays["image_ids"] = np.asarray(["image_001.png"], dtype=str)
        arrays["hierarchical_medoid_ids"] = np.asarray(["image_001.png"], dtype=str)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            write_deterministic_npz(first, arrays)
            write_deterministic_npz(second, arrays)
            self.assertEqual(sha256_file(first), sha256_file(second))
            with np.load(first, allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), set(ARRAY_CONTRACT))
                self.assertEqual(loaded["image_ids"].tolist(), ["image_001.png"])

    def test_plot_projection_alignment_flips_components_not_reference_points(self) -> None:
        computed = np.asarray([[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]])
        target = computed * np.asarray([-1.0, 1.0])
        arrays = {
            "plot_reference_coordinates": computed,
            "plot_pca_components": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "kmeans_centroids": np.asarray([[2.0, 4.0], [-3.0, 5.0]]),
        }
        aligned, metadata = align_plot_projection_to_static(
            arrays,
            target,
            context="fixture",
            residual_tolerance=1e-12,
            align_kmeans_centroids=True,
        )
        np.testing.assert_array_equal(aligned["plot_reference_coordinates"], target)
        np.testing.assert_array_equal(
            aligned["plot_pca_components"],
            np.asarray([[-1.0, -2.0], [3.0, 4.0]]),
        )
        self.assertEqual(metadata["plot_axis_signs"], [-1, 1])
        np.testing.assert_array_equal(
            aligned["kmeans_centroids"],
            np.asarray([[-2.0, 4.0], [3.0, 5.0]]),
        )

    def test_fitted_arrays_are_deterministic(self) -> None:
        rng = np.random.default_rng(1234)
        voxels = rng.normal(size=(12, 8)).astype(np.float64)
        source = ValidatedSource(
            candidate=SourceCandidate(
                model="fixture",
                roi="ffa",
                timestamp="20260101000000000Z",
                path=Path("fixture.csv"),
            ),
            image_ids=tuple(f"image_{index:03d}.png" for index in range(1, 13)),
            voxel_columns=tuple(f"v{index}" for index in range(8)),
            voxels=voxels,
            means=voxels.mean(axis=1),
            sha256="0" * 64,
        )
        first, first_metadata = fit_reference_arrays(source, pca_dim=5, k_max=3)
        second, second_metadata = fit_reference_arrays(source, pca_dim=5, k_max=3)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)


class ReferenceArtifactRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_dir = REPO_ROOT / "data" / "image_by_model"
        cls.combined_path = REPO_ROOT / "public" / "data" / "combined_image_means.csv"
        cls.candidates = discover_source_candidates(cls.raw_dir)
        cls.models = sorted({model for model, _ in cls.candidates})
        cls.combined = load_combined_means(cls.combined_path, cls.models)

        candidate = cls.candidates[("clip_rn50", "ffa")][0]
        cls.clip_source = load_and_validate_source(
            candidate,
            expected_ids=canonical_image_ids(),
            expected_voxel_count=716,
            expected_means=cls.combined["clip_rn50_ffa"].to_numpy(dtype=np.float64),
        )

    def test_complete_26_model_triads_are_discovered(self) -> None:
        self.assertEqual(len(self.models), 26)
        self.assertEqual(len(self.candidates), 78)
        self.assertTrue(all(len(candidates) >= 1 for candidates in self.candidates.values()))

    def test_raw_means_match_checked_aggregate(self) -> None:
        expected = self.combined["clip_rn50_ffa"].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            self.clip_source.means,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_voxel_cluster_labels_match_current_static_outputs(self) -> None:
        arrays, _ = fit_reference_arrays(self.clip_source)
        kmeans = pd.read_csv(
            REPO_ROOT
            / "public"
            / "data"
            / "voxel_clusters"
            / "murtylab_clip_rn50_nsd_1000_ffa_20260408164117065Z_clusters.csv"
        )
        hierarchical = pd.read_csv(
            REPO_ROOT
            / "public"
            / "data"
            / "voxel_clusters_hierarchical_corr"
            / "murtylab_clip_rn50_nsd_1000_ffa_20260408164117065Z_clusters.csv"
        )
        self.assertEqual(arrays["image_ids"].tolist(), kmeans["image_name"].tolist())
        np.testing.assert_array_equal(arrays["kmeans_labels"], kmeans["cluster_label"])
        np.testing.assert_array_equal(
            arrays["hierarchical_labels"],
            hierarchical["cluster_label"],
        )
        np.testing.assert_allclose(
            arrays["plot_reference_coordinates"],
            kmeans[["pca_x", "pca_y"]].to_numpy(dtype=np.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_visual_cluster_labels_match_current_static_outputs(self) -> None:
        feature_names, features, _ = extract_and_validate_visual_features(
            image_dir=REPO_ROOT / "public" / "images",
            checked_features_path=REPO_ROOT
            / "public"
            / "data"
            / "visual_clusters"
            / "visual_features_only.csv",
            expected_ids=canonical_image_ids(),
        )
        arrays, metadata = fit_visual_reference_arrays(
            image_ids=canonical_image_ids(),
            feature_names=feature_names,
            reference_features=features,
        )
        current_kmeans = pd.read_csv(
            REPO_ROOT / "public" / "data" / "visual_clusters" / "visual_clusters.csv"
        )
        current_hierarchical = pd.read_csv(
            REPO_ROOT
            / "public"
            / "data"
            / "visual_clusters_hierarchical_corr"
            / "visual_clusters.csv"
        )
        self.assertEqual(metadata["feature_count"], 77)
        np.testing.assert_array_equal(arrays["kmeans_labels"], current_kmeans["cluster_label"])
        np.testing.assert_array_equal(
            arrays["hierarchical_labels"],
            current_hierarchical["cluster_label"],
        )

    def test_cross_repository_supported_model_subset(self) -> None:
        report = validate_registries(
            raw_dir=self.raw_dir,
            core_registry=REPO_ROOT.parent
            / "cortex-core"
            / "cortex"
            / "models"
            / "__init__.py",
            app_constants=REPO_ROOT.parent
            / "cortex-app"
            / "assets"
            / "js"
            / "constants.jsx",
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["counts"]["infoviz_reference"], 26)
        self.assertEqual(report["strict_subset_errors"]["missing_from_core"], [])
        self.assertEqual(report["strict_subset_errors"]["missing_from_app"], [])

    def test_manifest_schema_is_valid_json(self) -> None:
        schema_path = REPO_ROOT / "scripts" / "data" / "reference_manifest.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

        generated_manifest_path = REPO_ROOT / "artifacts" / "reference" / "manifest.json"
        if generated_manifest_path.exists():
            try:
                import jsonschema
            except ImportError:
                self.skipTest("jsonschema is not installed")
            jsonschema.validate(
                json.loads(generated_manifest_path.read_text(encoding="utf-8")),
                schema,
            )

    def test_public_manifest_has_required_optional_capabilities(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / "public" / "data" / "reference_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["reference_count"], 185)
        self.assertTrue(manifest["optional_artifacts"]["visual"]["available"])
        self.assertEqual(
            manifest["optional_artifacts"]["visual"]["feature_schema"],
            "infoviz_visual_77_v1",
        )
        self.assertFalse(manifest["optional_artifacts"]["dreamsim"]["available"])

    def test_full_bundle_coordinates_match_all_checked_static_axes(self) -> None:
        artifact_root = REPO_ROOT / "artifacts" / "reference"
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("Full generated bundle is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked = 0
        for model, rois in manifest["artifacts"].items():
            for roi, entry in rois.items():
                with np.load(artifact_root / entry["path"], allow_pickle=False) as artifact:
                    static = pd.read_csv(
                        REPO_ROOT
                        / "public"
                        / "data"
                        / "voxel_clusters"
                        / f"{Path(entry['source_file']).stem}_clusters.csv"
                    )
                    np.testing.assert_array_equal(
                        artifact["plot_reference_coordinates"],
                        static[["pca_x", "pca_y"]].to_numpy(dtype=np.float64),
                        err_msg=f"{model}/{roi}",
                    )
                self.assertIn("plot_axis_signs", entry)
                checked += 1
        self.assertEqual(checked, 78)

        visual_entry = manifest["optional_artifacts"]["visual"]
        with np.load(artifact_root / visual_entry["path"], allow_pickle=False) as artifact:
            static_visual = pd.read_csv(
                REPO_ROOT / "public" / "data" / "visual_clusters" / "visual_clusters.csv"
            )
            np.testing.assert_array_equal(
                artifact["plot_reference_coordinates"],
                static_visual[["plot_x", "plot_y"]].to_numpy(dtype=np.float64),
            )
            distances = np.linalg.norm(
                artifact["plot_reference_coordinates"][:, np.newaxis, :]
                - artifact["kmeans_centroids"][np.newaxis, :, :],
                axis=2,
            )
            np.testing.assert_array_equal(
                np.argmin(distances, axis=1),
                artifact["kmeans_labels"],
            )


if __name__ == "__main__":
    unittest.main()
