from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.data import verify_live_coverage as qa


def ranking_fields(values, uploaded_means):
    """Supply genuine production-shaped statistics for the independent QA gate."""
    values = np.asarray(values, dtype=np.float64)
    # Fixture uses lexsort rather than QA's tuple sort for stable combined ties.
    all_values = np.concatenate((values, uploaded_means))
    ordered = np.lexsort((np.arange(len(all_values)), -all_values))
    ranks = np.empty(len(all_values), dtype=np.int64)
    ranks[ordered] = np.arange(1, len(all_values) + 1)
    results = []
    for index, mean in enumerate(uploaded_means):
        greater, equal, less = (int(np.sum(values > mean)), int(np.sum(values == mean)), int(np.sum(values < mean)))
        rank = int(ranks[len(values) + index])
        closest = np.lexsort((np.arange(len(values)), np.abs(values - mean)))[:5]
        results.append({
            "mean": float(mean), "reference_rank": float(greater + (equal + 1) / 2 if equal else greater + 1),
            "reference_percentile": 100.0 * (less + equal / 2) / len(values),
            "session_rank": rank, "session_count": len(all_values),
            "enters_top": rank <= min(10, len(values)),
            "enters_bottom": rank > len(all_values) - min(10, len(values)),
            "closest_by_mean": [{"image_name": qa.REFERENCE_IDS[i]} for i in closest],
        })
    return results


def joint_tree(stimulus_ids):
    """A complete valid one-cluster average-linkage correlation tree fixture."""
    ids = [("reference", image) for image in qa.REFERENCE_IDS] + [("uploaded", stimulus) for stimulus in stimulus_ids]
    n = len(ids)
    linkage = [[0, 1, 0.0, 2]]
    for leaf in range(2, n):
        linkage.append([n + leaf - 2, leaf, leaf / n, leaf + 1])
    return {
        **qa.JOINT_METADATA, "reference_count": len(qa.REFERENCE_IDS), "uploaded_count": len(stimulus_ids),
        "cluster_count": 1, "silhouette": None, "medoid_indices": [0], "linkage": linkage,
        "members": [{"kind": kind, "id": value, "cluster_id": 0, "distance_to_medoid": 0.0 if index == 0 else .1}
                    for index, (kind, value) in enumerate(ids)],
    }


class FakeResponse:
    def __init__(self, payload=None, image=False):
        self.payload = payload
        self.headers = {"content-type": "image/png" if image else "application/json"}

    def raise_for_status(self):
        pass

    def json(self):
        return copy.deepcopy(self.payload)


class FakeHTTP:
    def __init__(self, capabilities, sessions):
        self.capabilities = capabilities
        self.sessions = sessions
        self.posts = []
        self.closed = False

    def get(self, url, **kwargs):
        if url.endswith("/comparison/capabilities"):
            return FakeResponse(self.capabilities)
        if "/images/" in url:
            return FakeResponse(image=True)
        model = url.rsplit("/", 1)[-1].removeprefix("local-session-")
        return FakeResponse(self.sessions[model])

    def post(self, url, json, **kwargs):
        self.posts.append(copy.deepcopy(json))
        return FakeResponse({"session_id": f"local-session-{json['model']}"})

    def close(self):
        self.closed = True


class LiveCoverageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "artifacts/reference").mkdir(parents=True)
        (self.root / "public/images").mkdir(parents=True)
        for image_id in qa.IMAGE_IDS:
            (self.root / "public/images" / image_id).write_bytes(b"copied fixture image")
        self.reference = {
            "dataset": "nsd_1000", "reference_count": 185, "reference_version": "fixture-version",
            "supported_models": ["modela", "modelb"],
            "coverage_extension": {"added_models": ["modela", "modelb"], "model_prediction_policies": {}},
            "artifacts": {model: {roi: {"source_file": f"{model}_{roi}.csv", "path": f"{model}_{roi}.npz"} for roi in qa.ROIS}
                          for model in ("modela", "modelb")},
        }
        rng = np.random.default_rng(42)
        widths = {"ffa": 716, "ppa": 753, "eba": 2045}
        self.frames = {roi: pd.DataFrame(rng.normal(size=(185, widths[roi])).astype(np.float32),
                                        index=qa.REFERENCE_IDS) for roi in qa.ROIS}
        self.reference_means = {roi: frame.to_numpy().mean(axis=1, dtype=np.float64) for roi, frame in self.frames.items()}
        for model, entries in self.reference["artifacts"].items():
            for roi, entry in entries.items():
                path = self.root / "artifacts/reference" / entry["path"]
                np.savez(path, image_ids=np.array(qa.REFERENCE_IDS), reference_means=self.reference_means[roi])
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_reference()
        self.capabilities = {
            "available": True, "reference_version": "fixture-version", "reference_count": 185,
            "supported_models": ["modela", "modelb"], "dataset": "nsd_1000",
            "comparison_dataset": "nsd_1000", "rois": list(qa.ROIS),
        }
        self.sessions = {model: self.make_session(model) for model in ("modela", "modelb")}
        self.http = FakeHTTP(self.capabilities, self.sessions)
        self.addCleanup(patch.stopall)
        patch.object(qa, "REPO_ROOT", self.root).start()
        patch.object(qa.requests, "Session", return_value=self.http).start()
        patch.object(qa.pd, "read_csv", side_effect=self.read_csv).start()

    def write_reference(self):
        (self.root / "artifacts/reference/manifest.json").write_text(json.dumps(self.reference))

    def read_csv(self, path, **kwargs):
        return self.frames[Path(path).stem.rsplit("_", 1)[-1]].copy()

    def make_session(self, model):
        ids = [f"{model}-stimulus-{index}" for index in range(len(qa.IMAGE_IDS))]
        brain = {
            "subject_id": "s1", "response_space": "nsd_1000_subject1_floc_t7_v1",
            "roi_voxel_counts": qa.COUNTS,
            "stimuli": [{"stimulus_id": stimulus_id, "roi_responses": {}} for stimulus_id in ids],
        }
        roi_results = {}
        for roi in qa.ROIS:
            matrix = self.frames[roi].loc[list(qa.IMAGE_IDS)].to_numpy(dtype=np.float32)
            statistics = ranking_fields(self.reference_means[roi], matrix.mean(axis=1, dtype=np.float64))
            for index, stimulus in enumerate(brain["stimuli"]):
                stimulus["roi_responses"][roi] = matrix[index, :qa.COUNTS[roi]].tolist()
            roi_results[roi] = {
                "voxel_count": matrix.shape[1], "rdm": qa.normalized_euclidean_rdm(matrix).tolist(),
                "stimuli": [{"stimulus_id": stimulus_id, **statistics[index],
                             "sem": 0.1, "closest_in_voxel_space": [{"image_name": qa.IMAGE_IDS[index]}]}
                            for index, stimulus_id in enumerate(ids)],
                "joint_hierarchy": joint_tree(ids),
            }
        return {
            "schema_version": 1, "source_request": {"model": model, "dataset": "nsd_1000", "roi": "ffa"},
            "comparison": {"model": model, "dataset": "nsd_1000", "rois": list(qa.ROIS),
                           "reference_count": 185, "reference_version": "fixture-version"},
            "stimuli": [{"id": stimulus_id, "display_name": qa.IMAGE_IDS[index],
                         "image_url": f"/comparison-sessions/local-session-{model}/images/{index}"}
                        for index, stimulus_id in enumerate(ids)],
            "brain_map_results": brain, "roi_results": roi_results,
        }

    def run_qa(self, models=None):
        with redirect_stdout(io.StringIO()):
            return qa.verify_live(models or ["modela"], "http://mock.invalid", self.root / "uploads")

    def test_success_checks_all_rois_full_rdm_and_sequential_session_identity(self):
        report = self.run_qa(["modela", "modelb"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual([r["model"] for r in report["checked_models"]], ["modela", "modelb"])
        self.assertEqual([r["model"] for r in self.http.posts], ["modela", "modelb"])
        self.assertEqual(report["checked_models"][0]["maximum_full_pooled_rdm_drift"], 0)
        self.assertEqual(report["golden_tolerance"], 5e-4)
        self.assertTrue(report["checked_models"][0]["ranking_statistics_checked"])
        self.assertTrue(report["checked_models"][0]["joint_tree_contract_checked"])
        self.assertTrue(self.http.closed)
        self.assertEqual(list((self.root / "uploads").iterdir()), [])

    def test_exact_capabilities_are_required_before_any_session(self):
        for key, value in (("available", False), ("reference_count", 184), ("dataset", "nsd_500"),
                           ("comparison_dataset", "nsd_500"), ("rois", ["ffa", "eba", "ppa"])):
            with self.subTest(key=key):
                self.http.capabilities = {**self.capabilities, key: value}
                with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
                    self.run_qa()
                self.assertEqual(raised.exception.report["checked_models"], [])
                self.assertEqual(self.http.posts, [])

    def test_roi_and_brain_stimulus_order_must_match_image_identity(self):
        for location in ("stimuli", "brain", "roi", "hierarchy"):
            with self.subTest(location=location):
                session = self.make_session("modela")
                if location == "stimuli":
                    session["stimuli"].reverse()
                elif location == "brain":
                    session["brain_map_results"]["stimuli"].reverse()
                elif location == "roi":
                    session["roi_results"]["ffa"]["stimuli"].reverse()
                else:
                    session["roi_results"]["ffa"]["joint_hierarchy"]["members"][-1]["id"] = "wrong-upload"
                self.http.sessions["modela"] = session
                with self.assertRaises(qa.LiveCoverageVerificationError):
                    self.run_qa()

    def test_full_pooled_rdm_drift_fails_even_with_unchanged_subject1_and_means(self):
        result = self.http.sessions["modela"]["roi_results"]["ffa"]
        tolerance = qa.rdm_absolute_tolerance(result["voxel_count"])
        result["rdm"][0][1] += 3 * tolerance
        result["rdm"][1][0] += 3 * tolerance
        with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
            self.run_qa()
        self.assertIn("full pooled response RDM drift", raised.exception.report["error"])

    def test_joint_tree_requires_explicit_method_and_normalization_metadata(self):
        for field in qa.JOINT_METADATA:
            with self.subTest(field=field):
                self.http.sessions["modela"] = self.make_session("modela")
                self.http.sessions["modela"]["roi_results"]["ffa"]["joint_hierarchy"].pop(field)
                with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
                    self.run_qa()
                self.assertIn("method metadata", raised.exception.report["error"])
        self.http.sessions["modela"] = self.make_session("modela")
        self.http.sessions["modela"]["roi_results"]["ffa"]["joint_hierarchy"]["normalization"] = "joint_zscore"
        with self.assertRaises(qa.LiveCoverageVerificationError):
            self.run_qa()

    def test_core_tree_validator_rejects_broken_linkage_and_medoid_contract(self):
        for mutation in ("missing-linkage", "short-linkage", "bad-child", "bad-count", "reused-child", "bad-medoid"):
            with self.subTest(mutation=mutation):
                self.http.sessions["modela"] = self.make_session("modela")
                joint = self.http.sessions["modela"]["roi_results"]["ffa"]["joint_hierarchy"]
                if mutation == "missing-linkage":
                    joint.pop("linkage")
                elif mutation == "short-linkage":
                    joint["linkage"].pop()
                elif mutation == "bad-child":
                    joint["linkage"][0][0] = 188
                elif mutation == "bad-count":
                    joint["linkage"][-1][3] = 187
                elif mutation == "reused-child":
                    joint["linkage"][1][0] = 0
                else:
                    joint["members"][0]["distance_to_medoid"] = .1
                with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
                    self.run_qa()
                self.assertIn("tree/cluster contract", raised.exception.report["error"])

    def test_live_ranking_fields_and_neighbor_order_are_mandatory(self):
        for field in ("reference_rank", "reference_percentile", "session_rank", "session_count",
                      "enters_top", "enters_bottom", "closest_by_mean"):
            with self.subTest(field=field):
                self.http.sessions["modela"] = self.make_session("modela")
                stimulus = self.http.sessions["modela"]["roi_results"]["ffa"]["stimuli"][0]
                if field == "closest_by_mean":
                    stimulus[field].reverse()
                elif field.startswith("enters_"):
                    stimulus[field] = not stimulus[field]
                else:
                    stimulus[field] += 1
                with self.assertRaises(qa.LiveCoverageVerificationError):
                    self.run_qa()
                stimulus.pop(field)
                with self.assertRaises(qa.LiveCoverageVerificationError):
                    self.run_qa()

    def test_private_saved_means_are_checksum_bound_and_image_aligned(self):
        entry = self.reference["artifacts"]["modela"]["ffa"]
        path = self.root / "artifacts/reference" / entry["path"]
        np.savez(path, image_ids=np.array(qa.REFERENCE_IDS), reference_means=self.reference_means["ffa"] + 1)
        with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
            self.run_qa()
        self.assertIn("checksum", raised.exception.report["error"])
        np.savez(path, image_ids=np.array(qa.REFERENCE_IDS[::-1]), reference_means=self.reference_means["ffa"])
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_reference()
        with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
            self.run_qa()
        self.assertIn("image identity/order", raised.exception.report["error"])

    def test_ranks_use_saved_means_not_recomputed_csv_voxel_averages(self):
        saved_means = self.reference_means["ffa"] + 1
        entry = self.reference["artifacts"]["modela"]["ffa"]
        path = self.root / "artifacts/reference" / entry["path"]
        np.savez(path, image_ids=np.array(qa.REFERENCE_IDS), reference_means=saved_means)
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_reference()
        stimuli = self.http.sessions["modela"]["roi_results"]["ffa"]["stimuli"]
        live_means = [stimulus["mean"] for stimulus in stimuli]
        for stimulus, statistics in zip(stimuli, ranking_fields(saved_means, live_means)):
            stimulus.update(statistics)
        self.assertEqual(self.run_qa()["status"], "passed")

    def test_numeric_rank_fields_reject_nonfinite_values_and_bool_coercion(self):
        values = np.arange(185, 0, -1, dtype=np.float64)
        for field, value in (("reference_rank", float("nan")), ("reference_percentile", float("inf")),
                             ("session_rank", True), ("session_count", 188.0), ("enters_top", 1),
                             ("mean", "177.0")):
            with self.subTest(field=field):
                stimuli = ranking_fields(values, [177., 177., 1.])
                stimuli[0][field] = value
                with self.assertRaises(ValueError):
                    qa.validate_ranking_statistics(values, stimuli, context="malformed")

    def test_reference_first_upload_ties_and_top_bottom_boundaries(self):
        values = np.arange(185, 0, -1, dtype=np.float64)
        stimuli = ranking_fields(values, [177., 177., 1.])
        self.assertEqual([s["session_rank"] for s in stimuli], [10, 11, 188])
        self.assertEqual([s["reference_rank"] for s in stimuli], [9., 9., 185.])
        self.assertEqual([s["enters_top"] for s in stimuli], [True, False, False])
        self.assertEqual([s["enters_bottom"] for s in stimuli], [False, False, True])
        qa.validate_ranking_statistics(values, stimuli, context="ties")
        stimuli[0]["session_rank"] = 9  # Upload must follow the equal reference.
        with self.assertRaisesRegex(ValueError, "session_rank"):
            qa.validate_ranking_statistics(values, stimuli, context="ties")

    def test_reference_midrank_percentile_and_equidistant_neighbors(self):
        values = np.array([3., 2., 2., 0.] + [-4.] * 181)
        stimuli = ranking_fields(values, [2., 1., 4.])
        self.assertEqual([s["reference_rank"] for s in stimuli], [2.5, 4., 1.])
        self.assertAlmostEqual(stimuli[0]["reference_percentile"], 100 * 183 / 185)
        self.assertEqual([n["image_name"] for n in stimuli[1]["closest_by_mean"]],
                         ["image_002.png", "image_003.png", "image_004.png", "image_001.png", "image_005.png"])
        qa.validate_ranking_statistics(values, stimuli, context="midranks")
        stimuli[0]["reference_rank"] = 2
        with self.assertRaisesRegex(ValueError, "reference_rank"):
            qa.validate_ranking_statistics(values, stimuli, context="midranks")

    def test_stochastic_ranks_use_returned_mean_not_a_regenerated_query_rank(self):
        self.standardize_fixture()
        result = self.http.sessions["modela"]["roi_results"]["ffa"]
        live_means = [100., 100., -100.]
        for stimulus, fields in zip(result["stimuli"], ranking_fields(self.reference_means["ffa"], live_means)):
            stimulus.update(fields)
        self.assertEqual(self.run_qa()["status"], "passed")
        result["stimuli"][1]["session_rank"] -= 1
        with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
            self.run_qa()
        self.assertIn("session_rank", raised.exception.report["error"])

    def test_stochastic_models_skip_draw_equality_but_keep_shape_and_identity_checks(self):
        self.reference["coverage_extension"]["model_prediction_policies"]["modela"] = {"response_policy": "seeded_stochastic_sampling"}
        self.write_reference()
        result = self.http.sessions["modela"]["roi_results"]["ffa"]
        result["rdm"][0][1] += 1
        result["rdm"][1][0] += 1
        self.http.sessions["modela"]["brain_map_results"]["stimuli"][0]["roi_responses"]["ffa"][0] += 1
        report = self.run_qa()
        self.assertTrue(report["checked_models"][0]["stochastic"])
        self.assertIsNone(report["checked_models"][0]["maximum_full_pooled_rdm_drift"])
        result["rdm"][0][1] = float("nan")
        with self.assertRaises(qa.LiveCoverageVerificationError):
            self.run_qa()

    def test_failure_preserves_previously_checked_records_and_failed_session_id(self):
        self.http.sessions["modelb"]["roi_results"]["ffa"]["stimuli"][0]["mean"] += 1
        with self.assertRaises(qa.LiveCoverageVerificationError) as raised:
            self.run_qa(["modela", "modelb"])
        report = raised.exception.report
        self.assertEqual(report["status"], "failed")
        self.assertEqual([r["model"] for r in report["checked_models"]], ["modela"])
        self.assertEqual(report["sessions"][-1], {"model": "modelb", "session_id": "local-session-modelb", "status": "failed"})
        self.assertEqual(report["failed_model"], "modelb")

    def test_cli_writes_failure_report_without_exposing_session_tokens(self):
        self.http.sessions["modelb"]["roi_results"]["ffa"]["stimuli"][0]["mean"] += 1
        report_path = self.root / "artifacts/coverage-expansion/qa-failure.json"
        argv = ["verify_live_coverage.py", "--models", "modela", "modelb", "--upload-root", str(self.root / "uploads"),
                "--report-path", str(report_path)]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            qa.main()
        self.assertEqual(raised.exception.code, 1)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["checked_models"][0]["session_id"], "local-session-modela")
        self.assertEqual(report["sessions"][-1]["session_id"], "local-session-modelb")
        self.assertNotIn("local-session-", stdout.getvalue())

    def test_model_selection_uses_published_added_roster_and_explicit_baseline(self):
        reference = {**self.reference, "supported_models": ["modela", "modelb", "clip_rn50"]}
        self.assertEqual(qa.select_qa_models(reference, None, True, ["clip_rn50"]), ["modela", "modelb", "clip_rn50"])
        self.assertEqual(qa.select_qa_models(reference, ["modelb"], False), ["modelb"])
        with self.assertRaises(ValueError):
            qa.select_qa_models(reference, ["modelb"], False, ["clip_rn50"])

    def test_all_supported_checks_every_model_in_published_order(self):
        self.assertEqual(qa.select_qa_models(self.reference, None, False, all_supported=True),
                         ["modela", "modelb"])
        for models, all_new, baseline in ((["modela"], False, None), (None, True, None),
                                          (None, False, ["modela"])):
            with self.subTest(models=models, all_new=all_new, baseline=baseline), self.assertRaises(ValueError):
                qa.select_qa_models(self.reference, models, all_new, baseline, all_supported=True)

    def standardize_fixture(self):
        self.reference.pop("coverage_extension")
        self.reference["prediction_standardization"] = {
            "policy": "full_roster_fp32_regeneration_v1",
            "regenerated_models": ["modela", "modelb"],
            "model_prediction_policies": {
                "modela": {"response_policy": "seeded_stochastic_sampling"},
                "modelb": {"response_policy": "evaluation_mode"},
            },
        }
        self.write_reference()

    def test_standardized_release_preserves_stochastic_noise_checks(self):
        self.standardize_fixture()
        self.http.sessions["modela"]["brain_map_results"]["stimuli"][0]["roi_responses"]["ffa"][0] += 1
        report = self.run_qa(["modela", "modelb"])
        self.assertEqual([row["stochastic"] for row in report["checked_models"]], [True, False])
        self.assertIsNone(report["checked_models"][0]["maximum_subject1_voxel_drift"])
        self.assertEqual(report["checked_models"][1]["maximum_subject1_voxel_drift"], 0)

    def test_standardized_release_rejects_missing_or_ambiguous_noise_policies(self):
        self.standardize_fixture()
        original = copy.deepcopy(self.reference)
        for mutation in ("missing-model", "wrong-roster", "wrong-policy", "dual-lineage"):
            self.reference = copy.deepcopy(original)
            standardization = self.reference["prediction_standardization"]
            if mutation == "missing-model":
                standardization["model_prediction_policies"].pop("modelb")
            elif mutation == "wrong-roster":
                standardization["regenerated_models"] = ["modela"]
            elif mutation == "wrong-policy":
                standardization["policy"] = "unreviewed"
            else:
                self.reference["coverage_extension"] = {}
            self.write_reference()
            with self.subTest(mutation=mutation), self.assertRaises(qa.LiveCoverageVerificationError):
                self.run_qa()
        self.assertEqual(self.http.posts, [])

    def test_cli_can_publish_standardization_audit_without_tokens_in_stdout(self):
        self.standardize_fixture()
        report_path = self.root / "artifacts/precision-standardization/live.json"
        argv = ["verify_live_coverage.py", "--all-supported", "--upload-root", str(self.root / "uploads"),
                "--report-path", str(report_path)]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
            qa.main()
        report = json.loads(report_path.read_text())
        self.assertEqual(len(report["checked_models"]), 2)
        self.assertNotIn("local-session-", stdout.getvalue())

    def test_rdm_matches_core_semantics_and_tolerance_derivation(self):
        matrix = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        expected = np.array([[0, .5, 1], [.5, 0, .5], [1, .5, 0]])
        np.testing.assert_array_equal(qa.normalized_euclidean_rdm(matrix), expected)
        self.assertEqual(qa.rdm_absolute_tolerance(4), qa.GOLDEN_TOLERANCE)
        with self.assertRaises(ValueError):
            qa.rdm_absolute_tolerance(0)


if __name__ == "__main__":
    unittest.main()
