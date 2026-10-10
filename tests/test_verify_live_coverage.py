from __future__ import annotations

import copy
from contextlib import redirect_stdout
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
            "artifacts": {model: {roi: {"source_file": f"{model}_{roi}.csv"} for roi in qa.ROIS}
                          for model in ("modela", "modelb")},
        }
        self.write_reference()
        rng = np.random.default_rng(42)
        widths = {"ffa": 716, "ppa": 753, "eba": 2045}
        self.frames = {roi: pd.DataFrame(rng.normal(size=(185, widths[roi])).astype(np.float32),
                                        index=qa.REFERENCE_IDS) for roi in qa.ROIS}
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
            for index, stimulus in enumerate(brain["stimuli"]):
                stimulus["roi_responses"][roi] = matrix[index, :qa.COUNTS[roi]].tolist()
            roi_results[roi] = {
                "voxel_count": matrix.shape[1], "rdm": qa.normalized_euclidean_rdm(matrix).tolist(),
                "stimuli": [{"stimulus_id": stimulus_id, "mean": float(matrix[index].mean(dtype=np.float64)),
                             "sem": 0.1, "closest_in_voxel_space": [{"image_name": qa.IMAGE_IDS[index]}]}
                            for index, stimulus_id in enumerate(ids)],
                "joint_hierarchy": {
                    "reference_count": 185, "uploaded_count": len(ids),
                    "members": [{"kind": "reference", "id": image_id} for image_id in qa.REFERENCE_IDS]
                               + [{"kind": "uploaded", "id": stimulus_id} for stimulus_id in ids],
                },
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

    def test_rdm_matches_core_semantics_and_tolerance_derivation(self):
        matrix = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        expected = np.array([[0, .5, 1], [.5, 0, .5], [1, .5, 0]])
        np.testing.assert_array_equal(qa.normalized_euclidean_rdm(matrix), expected)
        self.assertEqual(qa.rdm_absolute_tolerance(4), qa.GOLDEN_TOLERANCE)
        with self.assertRaises(ValueError):
            qa.rdm_absolute_tolerance(0)


if __name__ == "__main__":
    unittest.main()
