from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import install_coverage_release as installer


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class InstallCoverageReleaseTests(unittest.TestCase):
    """Exercise only filesystem transaction semantics; scientific verifier is mocked."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo = root / "repo"
        self.baseline = root / "baseline"
        self.stage = root / "stage"
        self.spatial = root / "spatial"
        self.raw = self.repo / "artifacts/coverage-expansion/raw"
        self.old_version = "a" * 64
        self.new_version = "b" * 64
        self.rois = ["ffa", "ppa", "eba"]
        old_reference = {"reference_version": self.old_version, "supported_models": ["old"]}
        write_json(self.repo / "artifacts/reference/manifest.json", old_reference)
        (self.repo / "artifacts/reference/old.npz").write_bytes(b"immutable old transform")
        write_json(self.repo / "public/data/reference_manifest.json", {"reference_version": self.old_version})
        write_json(self.repo / "public/data/brain/reference/manifest.json", {"referenceVersion": self.old_version})
        (self.repo / "public/data/brain/reference/old.f32").write_bytes(b"immutable old brain")
        write_json(self.repo / "public/data/voxel_selectivity_atlas/manifest.json", {"referenceVersion": self.old_version})
        (self.repo / "public/data/voxel_selectivity_atlas/old.csv").write_bytes(b"immutable old atlas")
        (self.repo / "public/data/scores.json").write_bytes(b"immutable old scores")
        (self.repo / "data/image_by_model").mkdir(parents=True)
        (self.repo / "data/image_by_model/old.csv").write_bytes(b"immutable old raw")
        for relative in ("artifacts/reference", "public/data", "data/image_by_model"):
            shutil.copytree(self.repo / relative, self.baseline / relative)
            if relative != "data/image_by_model":
                shutil.copytree(self.repo / relative, self.stage / relative)
        write_json(self.baseline / "snapshot.json", {
            "reference_version": self.old_version,
            "models": ["old"],
            "files": {relative: installer.digest(self.baseline / relative)
                      for relative in tree_bytes(self.baseline)},
        })
        self.new_source_names = {roi: f"murtylab_new_nsd_1000_{roi}_20261010123400000Z.csv" for roi in self.rois}
        for roi, name in self.new_source_names.items():
            path = self.raw / "new" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image,v1\nimage_001.png,{self.rois.index(roi) + 1}\n".encode())
        new_reference = {
            "reference_version": self.new_version, "supported_models": ["new", "old"], "rois": self.rois,
            "artifacts": {"new": {roi: {"source_file": name} for roi, name in self.new_source_names.items()}},
        }
        write_json(self.stage / "artifacts/reference/manifest.json", new_reference)
        (self.stage / "artifacts/reference/new.npz").write_bytes(b"validated new transform")
        write_json(self.stage / "public/data/reference_manifest.json", {"reference_version": self.new_version})
        for relative in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
            shutil.copytree(self.repo / relative, self.spatial / relative)
            write_json(self.spatial / relative / "manifest.json", {"referenceVersion": self.new_version})
        (self.spatial / "public/data/brain/reference/new.f32").write_bytes(b"validated new brain")
        (self.spatial / "public/data/voxel_selectivity_atlas/new.csv").write_bytes(b"validated new atlas")
        self.report = {
            "reference_version": self.new_version, "added_models": ["new"], "model_count": 2,
            "baseline_preserved": True, "live_install_performed": False,
        }
        self.before_reference = tree_bytes(self.repo / "artifacts/reference")
        self.before_data = tree_bytes(self.repo / "public/data")
        self.before_raw = tree_bytes(self.repo / "data/image_by_model")
        self.baseline_before = tree_bytes(self.baseline)
        self.stage_before = tree_bytes(self.stage)
        self.spatial_before = tree_bytes(self.spatial)
        self.raw_before = tree_bytes(self.raw)

    def install(self):
        return installer.install_release(self.stage, self.spatial, repo=self.repo,
                                         baseline=self.baseline, raw_root=self.raw)

    def assert_inputs_unchanged(self):
        self.assertEqual(tree_bytes(self.baseline), self.baseline_before)
        self.assertEqual(tree_bytes(self.stage), self.stage_before)
        self.assertEqual(tree_bytes(self.spatial), self.spatial_before)
        self.assertEqual(tree_bytes(self.raw), self.raw_before)

    def test_success_retains_exact_backups_and_appends_raw_without_overwriting(self):
        with patch.object(installer, "verify_release", return_value=dict(self.report)) as verifier:
            report = self.install()
        verifier.assert_called_once_with(self.stage, self.spatial, repo=self.repo,
                                         baseline=self.baseline, raw_root=self.raw)
        backup = self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}"
        self.assertTrue(report["live_install_performed"])
        self.assertEqual(report["backup_root"], str(backup))
        self.assertEqual(tree_bytes(backup / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(backup / "public/data"), self.before_data)
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), tree_bytes(self.stage / "artifacts/reference"))
        expected_data = tree_bytes(self.stage / "public/data")
        for relative, payload in tree_bytes(self.spatial / "public/data").items():
            expected_data[relative] = payload
        self.assertEqual(tree_bytes(self.repo / "public/data"), expected_data)
        expected_raw = dict(self.before_raw)
        for name in self.new_source_names.values():
            expected_raw[name] = (self.raw / "new" / name).read_bytes()
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), expected_raw)
        self.assertEqual(json.loads((backup / "install.json").read_text()), report)
        self.assertEqual(list((self.repo / "artifacts/coverage-expansion").glob(".release-install-*")), [])
        self.assert_inputs_unchanged()

    def test_mid_install_failure_restores_both_live_roots_and_retains_added_raw(self):
        original_rename = Path.rename
        def fail_second_install(path, destination):
            if ".release-install-" in str(path) and path.as_posix().endswith("/artifacts/reference"):
                raise OSError("injected second-directory installation failure")
            return original_rename(path, destination)
        with patch.object(installer, "verify_release", return_value=dict(self.report)), \
             patch.object(Path, "rename", new=fail_second_install):
            with self.assertRaisesRegex(OSError, "second-directory installation failure"):
                self.install()
        backup = self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}"
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.before_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        retained_raw = tree_bytes(backup / "added-raw")
        self.assertEqual(retained_raw, {name: (self.raw / "new" / name).read_bytes()
                                       for name in self.new_source_names.values()})
        self.assertTrue((backup / "failed-release/public/data/reference_manifest.json").is_file())
        self.assertFalse((backup / "install.json").exists())
        self.assertEqual(list((self.repo / "artifacts/coverage-expansion").glob(".release-install-*")), [])
        self.assert_inputs_unchanged()

    def test_partial_raw_copy_failure_retains_partial_and_complete_additions_offline(self):
        original_copy = shutil.copyfile
        failed_destination = self.repo / "data/image_by_model" / self.new_source_names["ppa"]
        def fail_after_partial_write(source, destination, *args, **kwargs):
            if Path(destination) == failed_destination:
                failed_destination.write_bytes(b"partial failed raw copy")
                raise OSError("injected partial raw copy failure")
            return original_copy(source, destination, *args, **kwargs)
        with patch.object(installer, "verify_release", return_value=dict(self.report)), \
             patch.object(installer.shutil, "copyfile", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "partial raw copy failure"):
                self.install()
        backup = self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}"
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.before_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        self.assertEqual(tree_bytes(backup / "added-raw"), {
            self.new_source_names["ffa"]: (self.raw / "new" / self.new_source_names["ffa"]).read_bytes(),
            self.new_source_names["ppa"]: b"partial failed raw copy",
        })
        self.assertFalse((backup / "install.json").exists())
        self.assertEqual(list((self.repo / "artifacts/coverage-expansion").glob(".release-install-*")), [])
        self.assert_inputs_unchanged()

    def test_existing_backup_refuses_replacement_before_any_live_write(self):
        backup = self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}"
        backup.mkdir()
        (backup / "keep.txt").write_bytes(b"prior backup")
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "backup already exists"):
                self.install()
        self.assertEqual((backup / "keep.txt").read_bytes(), b"prior backup")
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.before_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        self.assert_inputs_unchanged()

    def test_colliding_raw_source_is_not_overwritten_or_removed(self):
        collision = self.repo / "data/image_by_model" / self.new_source_names["ffa"]
        collision.write_bytes(b"existing user data")
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "collides with existing data"):
                self.install()
        self.assertEqual(collision.read_bytes(), b"existing user data")
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.before_data)
        self.assertFalse((self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}").exists())
        self.assert_inputs_unchanged()

    def test_duplicate_new_source_destination_refuses_install_before_any_copy(self):
        path = self.stage / "artifacts/reference/manifest.json"
        reference = json.loads(path.read_text())
        reference["artifacts"]["new"]["ppa"]["source_file"] = self.new_source_names["ffa"]
        write_json(path, reference)
        self.stage_before = tree_bytes(self.stage)
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "collides with existing data"):
                self.install()
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.before_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        self.assert_inputs_unchanged()

    def test_live_user_edit_is_preserved_and_prevents_install_even_with_same_version(self):
        (self.repo / "public/data/scores.json").write_bytes(b"user edited live data")
        expected_data = {**self.before_data, "scores.json": b"user edited live data"}
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "Live baseline changed during staging"):
                self.install()
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), expected_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        self.assertFalse((self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}").exists())
        self.assert_inputs_unchanged()

    def test_untracked_live_data_file_is_preserved_and_prevents_install(self):
        (self.repo / "public/data/user-notes.json").write_bytes(b"new user data")
        expected_data = {**self.before_data, "user-notes.json": b"new user data"}
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "Live baseline file coverage changed"):
                self.install()
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.before_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), expected_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.before_raw)
        self.assertFalse((self.repo / "artifacts/coverage-expansion" / f"pre-install-{self.new_version}").exists())
        self.assert_inputs_unchanged()

    def prepare_installed_candidate(self):
        """Retain the original baseline while simulating an installed candidate."""
        candidate = {
            "reference_version": "c" * 64, "supported_models": ["new", "old"],
            "coverage_extension": {"parent_reference_version": self.old_version},
            "artifacts": {"new": {}},
        }
        self.prior_names = {}
        for roi in self.rois:
            name = f"prior_new_{roi}.csv"
            self.prior_names[roi] = name
            path = self.repo / "data/image_by_model" / name
            path.write_bytes(f"previous candidate {roi}".encode())
            candidate["artifacts"]["new"][roi] = {
                "source_file": name, "source_sha256": installer.digest(path),
            }
        write_json(self.repo / "artifacts/reference/manifest.json", candidate)
        write_json(self.repo / "public/data/reference_manifest.json", {"reference_version": "c" * 64})
        path = self.stage / "artifacts/reference/manifest.json"
        expanded = json.loads(path.read_text())
        expanded["coverage_extension"] = {"parent_reference_version": self.old_version}
        write_json(path, expanded)
        self.replace_snapshot = self.repo / "artifacts/coverage-expansion/current-snapshot"
        write_json(self.replace_snapshot / "snapshot.json", {
            "reference_version": "c" * 64, "models": candidate["supported_models"],
            "files": {f"{relative}/{name}": installer.digest(self.repo / relative / name)
                      for relative in ("artifacts/reference", "public/data", "data/image_by_model")
                      for name in tree_bytes(self.repo / relative)},
        })
        self.candidate_reference = tree_bytes(self.repo / "artifacts/reference")
        self.candidate_data = tree_bytes(self.repo / "public/data")
        self.candidate_raw = tree_bytes(self.repo / "data/image_by_model")

    def replace_candidate(self):
        return installer.install_release(self.stage, self.spatial, repo=self.repo,
                                         baseline=self.baseline, raw_root=self.raw,
                                         replace_snapshot=self.replace_snapshot)

    def test_correction_preserves_prior_candidate_and_retires_only_its_added_raw(self):
        self.prepare_installed_candidate()
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            report = self.replace_candidate()
        backup = Path(report["backup_root"])
        self.assertEqual(report["replaced_reference_version"], "c" * 64)
        self.assertEqual(report["retained_prior_raw_sources"], 3)
        self.assertEqual(tree_bytes(backup / "artifacts/reference"), self.candidate_reference)
        self.assertEqual(tree_bytes(backup / "public/data"), self.candidate_data)
        self.assertEqual(tree_bytes(backup / "prior-added-raw"), {
            name: self.candidate_raw[name] for name in self.prior_names.values()})
        expected_raw = {"old.csv": self.before_raw["old.csv"], **{
            name: (self.raw / "new" / name).read_bytes() for name in self.new_source_names.values()}}
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), expected_raw)
        self.assertEqual(tree_bytes(self.baseline), self.baseline_before)

    def test_correction_failure_restores_prior_roots_and_retired_raw(self):
        self.prepare_installed_candidate()
        original_rename = Path.rename
        def fail_new_reference(path, destination):
            if ".release-install-" in str(path) and path.as_posix().endswith("/artifacts/reference"):
                raise OSError("injected correction failure")
            return original_rename(path, destination)
        with patch.object(installer, "verify_release", return_value=dict(self.report)), \
             patch.object(Path, "rename", new=fail_new_reference):
            with self.assertRaisesRegex(OSError, "correction failure"):
                self.replace_candidate()
        self.assertEqual(tree_bytes(self.repo / "artifacts/reference"), self.candidate_reference)
        self.assertEqual(tree_bytes(self.repo / "public/data"), self.candidate_data)
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.candidate_raw)
        self.assertEqual(tree_bytes(self.baseline), self.baseline_before)

    def test_correction_requires_same_roster_and_reviewed_lineage(self):
        for field, bad_value in (("supported_models", ["old"]),
                                 ("coverage_extension", {"parent_reference_version": "d" * 64})):
            with self.subTest(field=field):
                if not hasattr(self, "replace_snapshot"):
                    self.prepare_installed_candidate()
                path = self.repo / "artifacts/reference/manifest.json"
                valid = json.loads(path.read_text())
                write_json(path, {**valid, field: bad_value})
                with patch.object(installer, "verify_release", return_value=dict(self.report)):
                    with self.assertRaisesRegex(ValueError, "baseline lineage"):
                        self.replace_candidate()
                write_json(path, valid)
                self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.candidate_raw)

    def test_correction_never_retires_an_original_baseline_source(self):
        self.prepare_installed_candidate()
        path = self.repo / "artifacts/reference/manifest.json"
        current = json.loads(path.read_text())
        current["artifacts"]["new"]["ffa"] = {
            "source_file": "old.csv", "source_sha256": installer.digest(self.repo / "data/image_by_model/old.csv")}
        write_json(path, current)
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "Unsafe or changed prior"):
                self.replace_candidate()
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.candidate_raw)

    def test_correction_rejects_live_edit_after_its_exact_snapshot(self):
        self.prepare_installed_candidate()
        (self.repo / "public/data/scores.json").write_bytes(b"user edited candidate")
        with patch.object(installer, "verify_release", return_value=dict(self.report)):
            with self.assertRaisesRegex(ValueError, "Live baseline changed"):
                self.replace_candidate()
        self.assertEqual((self.repo / "public/data/scores.json").read_bytes(), b"user edited candidate")
        self.assertEqual(tree_bytes(self.repo / "data/image_by_model"), self.candidate_raw)


if __name__ == "__main__":
    unittest.main()
