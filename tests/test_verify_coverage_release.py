from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data.verify_coverage_release import child, digest, snapshot_baseline, verify_inference_provenance


class CoverageReleaseBoundaryTests(unittest.TestCase):
    def test_cli_forwards_explicit_prediction_stage(self):
        from scripts.data import verify_coverage_release as release
        argv = ["verify_coverage_release.py", "--stage-root", "reference-stage",
                "--spatial-stage-root", "spatial-stage", "--raw-root", "raw-fp32"]
        with patch("sys.argv", argv), patch.object(release, "verify_release", return_value={}) as verify:
            with patch("builtins.print"):
                release.main()
        verify.assert_called_once_with(Path("reference-stage"), Path("spatial-stage"),
                                       raw_root=Path("raw-fp32"))

    def fixture(self, root: Path) -> tuple[Path, Path]:
        repo = root / "repo"
        for relative in ("artifacts/reference", "public/data", "data/image_by_model"):
            (repo / relative).mkdir(parents=True)
            (repo / relative / "sample.bin").write_bytes(b"original payload")
        (repo / "artifacts/reference/manifest.json").write_text(json.dumps({
            "reference_version": "a" * 64, "supported_models": ["original"],
        }))
        return repo, root / "backup"

    def test_snapshot_preserves_every_byte_and_resumes_without_live_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, backup = self.fixture(Path(directory))
            report = snapshot_baseline(repo, backup)
            self.assertEqual(len(report["files"]), 4)
            for name, expected in report["files"].items():
                self.assertEqual(digest(repo / name), expected)
                self.assertEqual(digest(backup / name), expected)
            (repo / "public/data/sample.bin").write_bytes(b"subsequent user change")
            self.assertEqual(snapshot_baseline(repo, backup), report)

    def test_corrupt_existing_snapshot_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, backup = self.fixture(Path(directory))
            snapshot_baseline(repo, backup)
            target = backup / "public/data/sample.bin"
            target.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                snapshot_baseline(repo, backup)
            self.assertEqual(target.read_bytes(), b"corruption")

    def test_release_paths_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, _ = self.fixture(root)
            target = repo / "public/data/sample.bin"
            (repo / "link.bin").symlink_to(target)
            for name in ("../absent.bin", "link.bin", str(target.resolve())):
                with self.assertRaises(ValueError):
                    child(repo, name)

    def inference_fixture(self, root: Path):
        from scripts.data.extend_reference_coverage import FP32_NUMERIC_POLICY
        core = root / "core"
        helper = core / "cortex/utils/inference_precision.py"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"reviewed helper source")
        raw = root / "raw"
        proof = {
            "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": [],
            "numeric_policy": FP32_NUMERIC_POLICY, "precision_helper_sha256": digest(helper),
            "core_inference_source_sha256": {"cortex/utils/inference_precision.py": digest(helper)},
        }
        path = raw / "fbnet/provenance.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(proof))
        reference = {"coverage_extension": {
            "added_models": ["fbnet"], "new_inference_provenance_sha256": {"fbnet": digest(path)},
            "model_prediction_policies": {"fbnet": {
                "response_policy": "evaluation_mode", "deterministic_per_stimulus": True, "noise_modes": []}},
            "prediction_policies": {"fbnet": {
                "numeric_policy": FP32_NUMERIC_POLICY, "precision_helper_sha256": digest(helper)}},
        }}
        return reference, raw, core

    def test_release_rechecks_exact_fp32_proofs_and_helper_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, raw, core = self.inference_fixture(Path(directory))
            verify_inference_provenance(reference, raw, core)
            (core / "cortex/utils/inference_precision.py").write_bytes(b"changed helper source")
            with self.assertRaisesRegex(ValueError, "helper source changed"):
                verify_inference_provenance(reference, raw, core)

    def test_release_rejects_provenance_change_even_when_csv_did_not_change(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, raw, core = self.inference_fixture(Path(directory))
            path = raw / "fbnet/provenance.json"
            proof = json.loads(path.read_text())
            proof["numeric_policy"]["cudnn_allow_tf32"] = True
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "Inference provenance changed"):
                verify_inference_provenance(reference, raw, core)

    def test_release_requires_complete_precision_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, raw, core = self.inference_fixture(Path(directory))
            reference["coverage_extension"]["prediction_policies"] = {}
            with self.assertRaisesRegex(ValueError, "roster is incomplete"):
                verify_inference_provenance(reference, raw, core)


if __name__ == "__main__":
    unittest.main()
