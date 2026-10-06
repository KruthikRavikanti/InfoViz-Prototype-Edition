from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data.build_brain_reference_artifacts import (
    COUNTS,
    EXPECTED_REFERENCE_VERSION,
    OFFSETS,
    REPO_ROOT,
    ROIS,
    ROW_WIDTH,
    BrainReferenceValidationError,
    canonical_image_ids,
    load_audit_constants,
    validated_roi_matrix,
)


class BrainReferenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = REPO_ROOT / "public/data/brain/reference"
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())
        cls.source_manifest = json.loads((REPO_ROOT / "artifacts/reference/manifest.json").read_text())
        cls.audit = load_audit_constants(REPO_ROOT.parent / "cortex-core/cortex/comparison/brain_map_audit.py")

    def test_public_reference_identity_is_raw_predicted_subject1_not_measured_or_cluster_space(self):
        manifest = self.manifest
        self.assertEqual(manifest["referenceVersion"], EXPECTED_REFERENCE_VERSION)
        self.assertEqual(manifest["imageIds"], canonical_image_ids())
        self.assertEqual(manifest["dataset"], "nsd_1000")
        self.assertEqual(manifest["responseKind"], "model_prediction")
        self.assertEqual(manifest["subjectId"], "s1")
        self.assertEqual(manifest["normalization"], "none")
        self.assertEqual(manifest["dtype"], "float32-le")
        self.assertEqual(manifest["rowWidth"], ROW_WIDTH)
        self.assertEqual(set(manifest["models"]), set(self.audit["AUDITED_CHECKPOINT_SHA256"]))

    def test_all_26_model_payloads_exactly_preserve_raw_subject1_prefixes_and_hashes(self):
        for model, entry in self.manifest["models"].items():
            with self.subTest(model=model):
                payload = (self.output / entry["url"]).read_bytes()
                self.assertEqual(len(payload), 185 * ROW_WIDTH * 4)
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                matrix = np.frombuffer(payload, dtype="<f4").reshape(185, ROW_WIDTH)
                self.assertTrue(np.isfinite(matrix).all())
                for roi in ROIS:
                    source = self.source_manifest["artifacts"][model][roi]
                    with np.load(REPO_ROOT / "artifacts/reference" / source["path"], allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(
                            matrix[:, OFFSETS[roi]:OFFSETS[roi] + COUNTS[roi]],
                            arrays["reference_voxels"][:, :COUNTS[roi]],
                        )
                    provenance = entry["provenance"][roi]
                    self.assertEqual(provenance["artifactSha256"], source["sha256"])
                    self.assertEqual(provenance["sourceSha256"], source["source_sha256"])
                    self.assertEqual(provenance["voxelOrderSha256"], self.audit["AUDITED_VOXEL_ORDER_SHA256"][roi])
                    self.assertEqual(provenance["checkpointSha256"], self.audit["AUDITED_CHECKPOINT_SHA256"][model][ROIS.index(roi)])

    def test_audit_reader_never_executes_source_code_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.py"
            path.write_text("raise RuntimeError('must not execute')\nAUDITED_REFERENCE_VERSION = 'wrong'\n")
            with self.assertRaises(BrainReferenceValidationError):
                load_audit_constants(path)

    def test_exporter_rejects_changed_source_or_artifact_checksums_and_voxel_order(self):
        root = REPO_ROOT / "artifacts/reference"
        raw = REPO_ROOT / "data/image_by_model"
        entry = self.source_manifest["artifacts"]["clip_rn50"]["ffa"]
        expected_order = self.audit["AUDITED_VOXEL_ORDER_SHA256"]["ffa"]
        for changed, order in (
            ({**entry, "sha256": "0" * 64}, expected_order),
            ({**entry, "source_sha256": "0" * 64}, expected_order),
            (entry, "0" * 64),
        ):
            with self.subTest(changed=changed.get("sha256"), order=order):
                with self.assertRaises(BrainReferenceValidationError):
                    validated_roi_matrix(root, raw, changed, "ffa", order)
        validated = validated_roi_matrix(root, raw, entry, "ffa", expected_order)
        self.assertEqual(validated.shape, (185, 156))
        self.assertEqual(validated.dtype, np.dtype("<f4"))


if __name__ == "__main__":
    unittest.main()
