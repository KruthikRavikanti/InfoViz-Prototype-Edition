import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.compare_reference_releases import compare_arrays, compare_atlas, safe_child


class ReferenceReleaseComparisonTests(unittest.TestCase):
    def test_unchanged_responses_preserve_exact_ranks(self):
        matrix = np.array([[1, 2], [2, 3], [1, 2]], dtype=np.float32)
        report = compare_arrays(matrix, matrix.copy())
        self.assertEqual(report["maximum_absolute_voxel_change"], 0)
        self.assertEqual(report["mean_response_rank_spearman"], 1)
        self.assertEqual(report["images_with_changed_mean_response_midrank"], 0)

    def test_rank_changes_are_measured_not_hidden_by_small_numbers(self):
        before = np.array([[0, 0], [0.00001, 0.00001]], dtype=np.float32)
        after = before[::-1].copy()
        report = compare_arrays(before, after)
        self.assertAlmostEqual(report["mean_response_rank_spearman"], -1)
        self.assertEqual(report["images_with_changed_mean_response_midrank"], 2)

    def test_constant_to_nonconstant_correlation_is_explicitly_undefined(self):
        report = compare_arrays(np.zeros((2, 2)), np.array([[0, 0], [1, 1]]))
        self.assertIsNone(report["mean_response_rank_spearman"])

    def test_identical_constant_rank_vectors_still_have_undefined_correlation(self):
        report = compare_arrays(np.ones((3, 2)), np.ones((3, 2)))
        self.assertIsNone(report["mean_response_rank_spearman"])
        self.assertEqual(report["images_with_changed_mean_response_midrank"], 0)

    def test_saved_ranking_means_expose_changes_hidden_by_identical_voxel_bits(self):
        voxels = np.ones((2, 2), dtype=np.float32)
        report = compare_arrays(voxels, voxels, before_means=np.array([1 + 1e-9, 1]),
                                after_means=np.ones(2))
        self.assertEqual(report["maximum_absolute_voxel_change"], 0)
        self.assertEqual(report["maximum_absolute_raw_voxel_mean_change"], 0)
        self.assertGreater(report["maximum_absolute_mean_response_change"], 0)
        self.assertEqual(report["images_with_changed_mean_response_midrank"], 2)
        self.assertEqual(report["maximum_mean_response_midrank_change"], 0.5)
        self.assertIsNone(report["mean_response_rank_spearman"])

    def test_saved_ranking_means_require_complete_finite_aligned_pairs(self):
        voxels = np.ones((2, 2))
        for before, after in ((np.ones(2), None), (None, np.ones(2)),
                              (np.ones(1), np.ones(2)), (np.array([1, np.nan]), np.ones(2))):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                compare_arrays(voxels, voxels, before_means=before, after_means=after)

    def test_invalid_matrices_are_rejected(self):
        for matrix in (np.array([[np.nan]]), np.array([1]), np.empty((0, 1))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                compare_arrays(matrix, matrix)
        with self.assertRaises(ValueError):
            compare_arrays(np.ones((2, 2)), np.ones((2, 3)))

    def test_artifact_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("../outside.npz", "/tmp/outside.npz"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    safe_child(root, relative)
            self.assertEqual(safe_child(root, "models/a.npz"), root / "models/a.npz")

    def test_atlas_comparison_handles_label_permutations_and_checks_voxel_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"voxel_index": np.arange(844), "roi": ["ffa"] * 844,
                                  "roi_voxel_index": np.arange(844), "cluster_label": np.arange(844) % 14})
            frame.to_csv(root / "before.csv", index=False)
            after = frame.copy()
            after["cluster_label"] = (after["cluster_label"] + 3) % 14
            after.to_csv(root / "after.csv", index=False)
            self.assertEqual(compare_atlas(root / "before.csv", root / "after.csv"), 1)
            after = after.iloc[::-1]
            after.to_csv(root / "after.csv", index=False)
            with self.assertRaises(ValueError):
                compare_atlas(root / "before.csv", root / "after.csv")


if __name__ == "__main__":
    unittest.main()
