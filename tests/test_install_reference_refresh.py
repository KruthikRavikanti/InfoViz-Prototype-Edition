from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import install_reference_refresh as installer
import test_install_coverage_release as fixtures


class InstallReferenceRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo, self.parent, self.stage, self.spatial = (root / name for name in ("repo", "parent", "stage", "spatial"))
        self.raw = self.repo / "artifacts/precision-standardization/raw-fp32"
        self.old_version, self.new_version = "a" * 64, "b" * 64
        self.rois = ["ffa", "ppa", "eba"]
        current = {"reference_version": self.old_version, "supported_models": ["old"], "rois": self.rois}
        fixtures.write_json(self.repo / "artifacts/reference/manifest.json", current)
        (self.repo / "artifacts/reference/old.npz").write_bytes(b"prior reviewed transform")
        fixtures.write_json(self.repo / "public/data/reference_manifest.json", {"reference_version": self.old_version})
        (self.repo / "public/data/scores.json").write_bytes(b"unrelated scores")
        for relative in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
            fixtures.write_json(self.repo / relative / "manifest.json", {"referenceVersion": self.old_version})
            (self.repo / relative / "old.bin").write_bytes(b"prior reviewed spatial data")
            (self.repo / relative / "unrelated.txt").write_bytes(b"retain unrelated spatial note")
        (self.repo / "data/image_by_model").mkdir(parents=True)
        self.old_names = {roi: f"murtylab_old_nsd_1000_{roi}_20260408164117065Z.csv" for roi in self.rois}
        for roi, name in self.old_names.items():
            (self.repo / "data/image_by_model" / name).write_bytes(f"retained original {roi}".encode())
        (self.repo / "data/image_by_model/historical-orphan.csv").write_bytes(b"retain previous candidate raw too")
        for relative in installer.refresh.SNAPSHOT_ROOTS:
            shutil.copytree(self.repo / relative, self.parent / relative)
            if relative != "data/image_by_model":
                shutil.copytree(self.repo / relative, self.stage / relative)
        fixtures.write_json(self.parent / "snapshot.json", {
            "reference_version": self.old_version, "models": ["old"],
            "files": {relative: installer.digest(self.parent / relative) for relative in fixtures.tree_bytes(self.parent)},
        })
        self.new_names = {roi: f"murtylab_old_nsd_1000_{roi}_20261011000000000Z.csv" for roi in self.rois}
        entries = {}
        for roi, name in self.new_names.items():
            source = self.raw / "old" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"new uniform FP32 {roi}".encode())
            entries[roi] = {"source_file": name, "source_sha256": installer.digest(source)}
        self.reference = {
            "reference_version": self.new_version, "supported_models": ["old"], "rois": self.rois,
            "artifacts": {"old": entries},
            "prediction_standardization": {
                "policy": installer.refresh.POLICY, "parent_reference_version": self.old_version,
                "parent_manifest_sha256": installer.digest(self.parent / "artifacts/reference/manifest.json"),
                "parent_snapshot_sha256": installer.digest(self.parent / "snapshot.json"), "regenerated_models": ["old"],
            },
        }
        fixtures.write_json(self.stage / "artifacts/reference/manifest.json", self.reference)
        (self.stage / "artifacts/reference/old.npz").write_bytes(b"fresh complete transform")
        fixtures.write_json(self.stage / "public/data/reference_manifest.json", {"reference_version": self.new_version})
        for relative in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
            shutil.copytree(self.repo / relative, self.spatial / relative)
            fixtures.write_json(self.spatial / relative / "manifest.json", {"referenceVersion": self.new_version})
            (self.spatial / relative / "old.bin").write_bytes(b"fresh complete spatial data")
        self.report = {"policy": installer.refresh.POLICY, "reference_version": self.new_version,
                       "parent_reference_version": self.old_version, "regenerated_models": ["old"], "live_install_performed": False}
        self.before_roots = {relative: fixtures.tree_bytes(self.repo / relative) for relative in installer.refresh.SNAPSHOT_ROOTS}
        self.before_inputs = {path: fixtures.tree_bytes(path) for path in (self.parent, self.stage, self.spatial, self.raw)}

    def install(self):
        return installer.install_refresh_release(self.stage, self.spatial, parent_snapshot=self.parent,
                                                 repo=self.repo, raw_root=self.raw)

    def assert_inputs_unchanged(self):
        for path, before in self.before_inputs.items():
            self.assertEqual(fixtures.tree_bytes(path), before)

    def assert_original_live_state(self):
        for relative, before in self.before_roots.items():
            self.assertEqual(fixtures.tree_bytes(self.repo / relative), before)

    def test_success_backs_up_both_live_roots_and_keeps_every_historical_raw_in_place(self):
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)):
            report = self.install()
        self.assertEqual(report["historical_raw_sources_retired"], 0)
        self.assertEqual(report["appended_raw_sources"], 3)
        backup = Path(report["backup_root"])
        for relative in ("public/data", "artifacts/reference"):
            self.assertEqual(fixtures.tree_bytes(backup / relative), self.before_roots[relative])
        raw = fixtures.tree_bytes(self.repo / "data/image_by_model")
        for name, content in self.before_roots["data/image_by_model"].items():
            self.assertEqual(raw[name], content)
        for name in self.new_names.values():
            self.assertEqual(raw[name], (self.raw / "old" / name).read_bytes())
        self.assertEqual((self.repo / "public/data/brain/reference/old.bin").read_bytes(), b"fresh complete spatial data")
        self.assert_inputs_unchanged()

    def test_mid_install_failure_restores_both_roots_and_preserves_new_raw_offline(self):
        rename = Path.rename
        def fail_reference(path, destination):
            if ".reference-refresh-install-" in str(path) and path.as_posix().endswith("/artifacts/reference"):
                raise OSError("injected reference refresh failure")
            return rename(path, destination)
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), patch.object(Path, "rename", new=fail_reference):
            with self.assertRaisesRegex(OSError, "injected reference refresh failure"):
                self.install()
        self.assert_original_live_state()
        backup = self.repo / f"artifacts/precision-standardization/pre-install-{self.new_version}"
        self.assertEqual(set(fixtures.tree_bytes(backup / "added-raw")), set(self.new_names.values()))
        self.assertTrue((backup / "failed-release/public/data").is_dir())
        self.assert_inputs_unchanged()

    def test_partial_new_raw_copy_failure_retains_the_partial_file_and_restores_live_state(self):
        copyfile = shutil.copyfile
        fail_name = self.new_names["ppa"]
        def fail_copy(source, destination, *args, **kwargs):
            if Path(destination).parent == self.repo / "data/image_by_model" and Path(destination).name == fail_name:
                Path(destination).write_bytes(b"partial new copy")
                raise OSError("injected partial FP32 copy")
            return copyfile(source, destination, *args, **kwargs)
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), patch.object(installer.shutil, "copyfile", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "partial FP32 copy"):
                self.install()
        self.assert_original_live_state()
        backup = self.repo / f"artifacts/precision-standardization/pre-install-{self.new_version}/added-raw"
        self.assertEqual((backup / fail_name).read_bytes(), b"partial new copy")
        self.assert_inputs_unchanged()

    def test_a_collision_never_overwrites_any_historical_raw(self):
        name = self.new_names["ffa"]
        target = self.repo / "data/image_by_model" / name
        target.write_bytes(b"user existing data at new filename")
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), self.assertRaisesRegex(ValueError, "collides"):
            self.install()
        self.assertEqual(target.read_bytes(), b"user existing data at new filename")
        self.assertEqual(fixtures.tree_bytes(self.repo / "public/data"), self.before_roots["public/data"])

    def test_modified_or_untracked_live_data_prevents_install_without_losing_user_edits(self):
        path = self.repo / "public/data/scores.json"
        path.write_bytes(b"new user score edits")
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), self.assertRaisesRegex(ValueError, "Live parent changed"):
            self.install()
        self.assertEqual(path.read_bytes(), b"new user score edits")
        path.write_bytes(self.before_roots["public/data"]["scores.json"])
        (self.repo / "public/data/new-user-file.json").write_bytes(b"new user file")
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), self.assertRaisesRegex(ValueError, "file coverage changed"):
            self.install()
        self.assertEqual((self.repo / "public/data/new-user-file.json").read_bytes(), b"new user file")
        self.assertEqual(fixtures.tree_bytes(self.repo / "data/image_by_model"), self.before_roots["data/image_by_model"])

    def test_parent_digest_full_roster_and_policy_are_mandatory(self):
        path = self.stage / "artifacts/reference/manifest.json"
        variants = []
        for key, value in (("parent_manifest_sha256", "f" * 64), ("parent_snapshot_sha256", "f" * 64),
                           ("policy", "append_only_preserve_baseline_artifacts_v1"), ("regenerated_models", [])):
            variant = copy.deepcopy(self.reference)
            variant["prediction_standardization"][key] = value
            variants.append(variant)
        variants.append({**self.reference, "supported_models": ["old", "new"]})
        for variant in variants:
            fixtures.write_json(path, variant)
            with self.subTest(variant=variant), patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), self.assertRaisesRegex(ValueError, "exact reviewed installed parent"):
                self.install()
            self.assert_original_live_state()

    def test_existing_backup_is_not_replaced(self):
        backup = self.repo / f"artifacts/precision-standardization/pre-install-{self.new_version}"
        backup.mkdir()
        (backup / "important.txt").write_bytes(b"prior retained backup")
        with patch.object(installer, "verify_refresh_release", return_value=dict(self.report)), self.assertRaisesRegex(ValueError, "backup already exists"):
            self.install()
        self.assertEqual((backup / "important.txt").read_bytes(), b"prior retained backup")
        self.assert_original_live_state()


if __name__ == "__main__":
    unittest.main()
