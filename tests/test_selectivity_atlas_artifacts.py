from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data.build_selectivity_atlas_artifacts import (
    DATASET,
    DEFAULT_OUTPUT_DIR,
    K14_STABILITY_CAVEAT,
    AtlasConfig,
    AtlasFit,
    SelectivityAtlasValidationError,
    SubjectRoiSource,
    build_summary,
    combine_model_sources,
    evaluate_split_half_stability,
    fit_selectivity_atlas,
    load_subject1_csv,
    method_metadata,
    provenance_metadata,
    sha256_file,
    selectivity_features,
    validate_subject1_columns,
    write_points_csv,
)


class SelectivityAtlasArtifactTests(unittest.TestCase):
    def test_subject1_prefix_must_be_exact_contiguous_and_leading(self) -> None:
        columns = ["s1_0", "s1_1", "s1_2", "s2_0"]
        self.assertEqual(validate_subject1_columns(columns, 3), (0, 1, 2))

        invalid = (
            ["s1_0", "s1_2", "s1_1", "s2_0"],
            ["s2_0", "s1_0", "s1_1", "s1_2"],
            ["s1_0", "s1_1", "s1_2", "s1_3"],
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(
                SelectivityAtlasValidationError
            ):
                validate_subject1_columns(candidate, 3)

    def test_csv_loader_rejects_nonfinite_values_and_preserves_image_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "murtylab_fixture_nsd_1000_ffa_20260101000000000Z.csv"
            path.write_text(
                "image,s1_0,s1_1,s2_0\n"
                "image_001.png,1.0,2.0,9.0\n"
                "image_002.png,3.0,4.0,8.0\n",
                encoding="utf-8",
            )
            source = load_subject1_csv(
                path,
                model="fixture",
                roi="ffa",
                expected_count=2,
                expected_image_ids=("image_001.png", "image_002.png"),
            )
            np.testing.assert_array_equal(source.responses, [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(source.image_ids, ("image_001.png", "image_002.png"))

            path.write_text(
                "image,s1_0,s1_1,s2_0\n"
                "image_001.png,1.0,nan,9.0\n"
                "image_002.png,3.0,4.0,8.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SelectivityAtlasValidationError, "non-finite"):
                load_subject1_csv(
                    path,
                    model="fixture",
                    roi="ffa",
                    expected_count=2,
                    expected_image_ids=("image_001.png", "image_002.png"),
                )

    def test_selectivity_features_match_voxelwise_zscore_then_l2_norm(self) -> None:
        rng = np.random.default_rng(4)
        responses = rng.normal(size=(18, 9))
        features, z_profiles, explained = selectivity_features(responses, n_components=4)

        np.testing.assert_allclose(z_profiles.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(z_profiles.std(axis=0), 1.0, atol=2e-9)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=2e-9)
        self.assertEqual(features.shape, (9, 4))
        self.assertGreater(explained, 0)
        self.assertLessEqual(explained, 1)

    def test_full_fit_uses_display_embedder_after_kmeans(self) -> None:
        rng = np.random.default_rng(8)
        responses = rng.normal(size=(20, 12))
        config = AtlasConfig(
            pca_components=4,
            cluster_count=3,
            kmeans_n_init=7,
            kmeans_random_state=0,
        )
        calls: list[np.ndarray] = []

        def fake_embedder(features: np.ndarray, received: AtlasConfig) -> np.ndarray:
            self.assertEqual(received, config)
            calls.append(features.copy())
            return features[:, :3]

        fit = fit_selectivity_atlas(
            responses, config=config, embedder=fake_embedder
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(fit.labels.shape, (12,))
        self.assertEqual(fit.embedding.shape, (12, 3))
        self.assertEqual(set(fit.labels.tolist()), {0, 1, 2})
        self.assertTrue(np.isfinite(fit.silhouette))
        self.assertTrue(np.all(fit.distances_to_centroid >= 0))
        self.assertEqual(fit.split_half_stability["iterations"], 6)
        self.assertEqual(len(fit.split_half_stability["adjustedRandIndices"]), 6)

    def test_split_half_stability_is_deterministic_and_reports_thresholds(self) -> None:
        rng = np.random.default_rng(18)
        responses = rng.normal(size=(24, 15))
        config = AtlasConfig(
            pca_components=4,
            cluster_count=3,
            kmeans_n_init=5,
        )
        first = evaluate_split_half_stability(
            responses,
            config=config,
            iterations=3,
            random_seed=9,
            threshold=0.2,
        )
        second = evaluate_split_half_stability(
            responses,
            config=config,
            iterations=3,
            random_seed=9,
            threshold=0.2,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], 3)
        self.assertEqual(first["randomSeed"], 9)
        self.assertEqual(len(first["adjustedRandIndices"]), 3)
        self.assertEqual(
            first["meanPassesThreshold"], first["meanAdjustedRandIndex"] >= 0.2
        )
        self.assertEqual(
            first["allPassThreshold"],
            all(score >= 0.2 for score in first["adjustedRandIndices"]),
        )

    def test_summary_profiles_rank_preferred_and_suppressed_images(self) -> None:
        image_ids = ("a.png", "b.png", "c.png", "d.png")
        z_profiles = np.asarray(
            [
                [4.0, 2.0, -2.0, -4.0],
                [2.0, 1.0, -1.0, -2.0],
                [-2.0, -1.0, 1.0, 2.0],
                [-4.0, -2.0, 2.0, 4.0],
            ]
        )
        fit = AtlasFit(
            z_profiles=z_profiles,
            selectivity_features=np.ones((4, 2)),
            pca_explained_variance_ratio_sum=0.75,
            labels=np.asarray([0, 0, 1, 1], dtype=np.int32),
            embedding=np.zeros((4, 3)),
            distances_to_centroid=np.zeros(4),
            silhouette_by_voxel=np.asarray([0.2, 0.4, 0.3, 0.5]),
            silhouette=0.35,
            split_half_stability={
                "iterations": 6,
                "randomSeed": 42,
                "adjustedRandIndices": [0.8] * 6,
                "meanAdjustedRandIndex": 0.8,
                "minAdjustedRandIndex": 0.8,
                "threshold": 0.7,
                "meanPassesThreshold": True,
                "allPassThreshold": True,
            },
        )
        config = AtlasConfig(pca_components=2, cluster_count=2)
        summary = build_summary(
            model="fixture",
            image_ids=image_ids,
            roi_names=np.asarray(["ffa", "eba", "ppa", "ppa"]),
            fit=fit,
            source_files=(),
            config=config,
            top_image_count=2,
        )

        self.assertEqual(summary["dataset"], DATASET)
        self.assertEqual(summary["subject"], "s1")
        self.assertEqual(summary["stimulusSet"], "murty185")
        self.assertEqual(summary["responseSpace"], "nsd_1000_subject1_floc_t7_v1")
        self.assertEqual(summary["responseKind"], "model_predictions")
        self.assertFalse(summary["embeddingIsAnatomical"])
        self.assertEqual(summary["k14StabilityCaveat"], K14_STABILITY_CAVEAT)
        self.assertEqual(summary["splitHalfStability"]["iterations"], 6)
        self.assertEqual(summary["clusters"][0]["size"], 2)
        self.assertEqual(
            summary["clusters"][0]["roiCounts"], {"ffa": 1, "eba": 1, "ppa": 0}
        )
        self.assertEqual(
            summary["clusters"][0]["preferredImages"][0]["imageId"], "a.png"
        )
        self.assertEqual(
            summary["clusters"][0]["suppressedImages"][0]["imageId"], "d.png"
        )

    def test_points_csv_contract_preserves_global_and_roi_indices(self) -> None:
        fit = AtlasFit(
            z_profiles=np.zeros((3, 3)),
            selectivity_features=np.zeros((3, 2)),
            pca_explained_variance_ratio_sum=1.0,
            labels=np.asarray([2, 1, 0], dtype=np.int32),
            embedding=np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
            distances_to_centroid=np.asarray([0.1, 0.2, 0.3]),
            silhouette_by_voxel=np.zeros(3),
            silhouette=0.0,
            split_half_stability={},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            write_points_csv(
                path,
                roi_names=np.asarray(["ffa", "eba", "eba"]),
                roi_indices=np.asarray([0, 0, 1]),
                fit=fit,
            )
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(
            list(rows[0]),
            [
                "voxel_index",
                "roi",
                "roi_voxel_index",
                "cluster_label",
                "umap_x",
                "umap_y",
                "umap_z",
                "distance_to_centroid",
            ],
        )
        self.assertEqual(rows[2]["voxel_index"], "2")
        self.assertEqual(rows[2]["roi"], "eba")
        self.assertEqual(rows[2]["roi_voxel_index"], "1")

    def test_method_contract_states_umap_is_not_used_for_clustering(self) -> None:
        method = method_metadata(AtlasConfig())
        self.assertEqual(method["objectClustered"], "voxels")
        self.assertEqual(method["pcaComponents"], 35)
        self.assertEqual(method["clusterCount"], 14)
        self.assertEqual(method["kmeansNInit"], 25)
        self.assertFalse(method["embeddingUsedForClustering"])

    def test_provenance_contract_disambiguates_embedding_and_responses(self) -> None:
        self.assertEqual(
            provenance_metadata(),
            {
                "stimulusSet": "murty185",
                "responseSpace": "nsd_1000_subject1_floc_t7_v1",
                "responseKind": "model_predictions",
                "embeddingIsAnatomical": False,
            },
        )


@unittest.skipUnless(
    (DEFAULT_OUTPUT_DIR / "manifest.json").is_file(),
    "generated selectivity-atlas browser artifacts are unavailable",
)
class SelectivityAtlasLocalArtifactTests(unittest.TestCase):
    def test_public_bundle_is_complete_and_self_consistent(self) -> None:
        manifest = json.loads(
            (DEFAULT_OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["dataset"], "nsd_1000")
        self.assertEqual(manifest["subject"], "s1")
        self.assertEqual(manifest["stimulusSet"], "murty185")
        self.assertEqual(manifest["responseKind"], "model_predictions")
        self.assertFalse(manifest["embeddingIsAnatomical"])
        self.assertEqual(manifest["imageCount"], 185)
        self.assertEqual(manifest["voxelCount"], 844)
        self.assertEqual(manifest["clusterCount"], 14)
        self.assertEqual(manifest["roiOrder"], ["ffa", "eba", "ppa"])
        self.assertEqual(set(manifest["supportedModels"]), set(manifest["models"]))

        for model in manifest["supportedModels"]:
            with self.subTest(model=model):
                record = manifest["models"][model]
                points_path = DEFAULT_OUTPUT_DIR / record["pointsUrl"]
                summary_path = DEFAULT_OUTPUT_DIR / record["summaryUrl"]
                self.assertEqual(sha256_file(points_path), record["pointsSha256"])
                self.assertEqual(sha256_file(summary_path), record["summarySha256"])
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["model"], model)
                self.assertEqual(summary["subject"], manifest["subject"])
                self.assertEqual(summary["stimulusSet"], manifest["stimulusSet"])
                self.assertEqual(sum(item["size"] for item in summary["clusters"]), 844)
                self.assertEqual(summary["splitHalfStability"]["iterations"], 6)
                with points_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 844)
                self.assertEqual(int(rows[0]["voxel_index"]), 0)
                self.assertEqual(int(rows[-1]["voxel_index"]), 843)


if __name__ == "__main__":
    unittest.main()
