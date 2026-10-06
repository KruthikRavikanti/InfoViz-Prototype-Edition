from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.data.build_nsd_s1_brain_artifacts import (
    DEFAULT_ARCHIVE,
    DEFAULT_BODY_LABELS,
    DEFAULT_DERIVED_OUT,
    DEFAULT_MAPPER,
    DEFAULT_NSD_DATA_ROOT,
    DEFAULT_PUBLIC_OUT,
    DEFAULT_TENSOR_ROOT,
    EXPECTED_COORDINATE_SHA256,
    RESPONSE_SPACE,
    VOLUME_SHAPE_XYZ,
    VOLUME_SHAPE_ZYX,
    RoiSelection,
    load_projections,
    load_roi_selections,
    extract_surface_files,
    project_selection,
    projection_document,
    reconstruct_roi_selection,
    sha256_file,
    write_deterministic_npz,
)


class BrainArtifactUnitTests(unittest.TestCase):
    def test_selection_preserves_zyx_c_order_and_converts_xyz_linear_indices(self) -> None:
        cortical_mask = np.zeros(VOLUME_SHAPE_ZYX, dtype=np.bool_)
        source_coordinates = np.asarray(
            [[0, 0, 2], [0, 1, 0], [2, 0, 0]], dtype=np.int64
        )
        cortical_mask[tuple(source_coordinates.T)] = True
        labels = np.asarray([2, 2, 2], dtype=np.float64)
        t_values = np.asarray([8, 7, 9], dtype=np.float64)

        selection = reconstruct_roi_selection(
            name="fixture",
            cortical_mask_zyx=cortical_mask,
            roi_labels_1d=labels,
            t_values_1d=t_values,
            label_values=(2,),
        )

        # t == 7 is excluded, and the remaining coordinates retain C-order z/y/x.
        np.testing.assert_array_equal(
            selection.coordinates_zyx,
            np.asarray([[0, 0, 2], [2, 0, 0]], dtype=np.int16),
        )
        expected_linear = np.ravel_multi_index(
            ([2, 0], [0, 0], [0, 2]), VOLUME_SHAPE_XYZ, order="C"
        )
        np.testing.assert_array_equal(selection.linear_indices_xyz, expected_linear)

    def test_projection_uses_per_vertex_weights_and_tracks_column_coverage(self) -> None:
        selection = RoiSelection(
            name="fixture",
            coordinates_zyx=np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.int16),
            linear_indices_xyz=np.asarray([0, 1, 2], dtype=np.int32),
        )
        width = int(np.prod(VOLUME_SHAPE_XYZ))
        left = sparse.coo_matrix(
            (
                np.asarray([0.25, 0.75, 0.75]),
                (
                    np.asarray([0, 1, 0]),
                    np.asarray([0, 0, 1]),
                ),
            ),
            shape=(3, width),
        ).tocsr()
        right = sparse.coo_matrix(
            (
                np.asarray([0.25]),
                (np.asarray([2]), np.asarray([1])),
            ),
            shape=(4, width),
        ).tocsr()

        projection = project_selection(selection, {"lh": left, "rh": right})

        self.assertEqual(projection.hemispheres, ("lh", "bilateral", "unmapped"))
        self.assertEqual(projection.unmapped_response_indices, (2,))

        left_mapping = projection.mappings["lh"]
        # Vertex 0 receives responses 0 and 1 with raw weights .25 and .75.
        vertex_zero = left_mapping.vertex_indices == 0
        np.testing.assert_allclose(
            left_mapping.vertex_normalized_weights[vertex_zero], [0.25, 0.75]
        )
        # Vertex 1 receives only response 0, irrespective of raw weight .75.
        np.testing.assert_allclose(
            left_mapping.vertex_normalized_weights[left_mapping.vertex_indices == 1], [1.0]
        )
        # The ignored forensic normalization still spans both hemispheres per response.
        response_one_left = left_mapping.response_indices == 1
        np.testing.assert_allclose(
            left_mapping.response_normalized_weights[response_one_left], [0.75]
        )
        np.testing.assert_allclose(
            projection.mappings["rh"].response_normalized_weights, [0.25]
        )

    def test_projection_document_has_fail_closed_compatibility_contract(self) -> None:
        selection = RoiSelection(
            name="fixture",
            coordinates_zyx=np.asarray([[0, 0, 0]], dtype=np.int16),
            linear_indices_xyz=np.asarray([0], dtype=np.int32),
        )
        width = int(np.prod(VOLUME_SHAPE_XYZ))
        left = sparse.coo_matrix(([1.0], ([0], [0])), shape=(1, width)).tocsr()
        right = sparse.csr_matrix((1, width), dtype=np.float64)
        document = projection_document(
            {"ffa": project_selection(selection, {"lh": left, "rh": right})}
        )

        self.assertEqual(document["schemaVersion"], 1)
        self.assertEqual(document["subjectId"], "s1")
        self.assertEqual(document["responseSpace"], RESPONSE_SPACE)
        self.assertEqual(document["fLocThreshold"], 7)
        self.assertEqual(document["fLocThresholdOperator"], ">")
        self.assertEqual(document["responseOrdering"], "zyx-c-order")
        self.assertEqual(document["mapper"], "linenn")
        self.assertEqual(document["volumeShapeXyz"], [83, 104, 81])
        self.assertEqual(document["normalizationPolicy"], "per-surface-vertex")
        self.assertEqual(document["rois"]["ffa"]["responseCount"], 1)
        self.assertEqual(document["rois"]["ffa"]["mappedResponseCount"], 1)

    def test_deterministic_npz_has_stable_hash(self) -> None:
        arrays = {
            "coordinates": np.asarray([[1, 2, 3]], dtype=np.int16),
            "weights": np.asarray([0.25, 0.75], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            write_deterministic_npz(first, arrays)
            write_deterministic_npz(second, arrays)
            self.assertEqual(sha256_file(first), sha256_file(second))
            with np.load(first, allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), set(arrays))


LOCAL_SOURCES_AVAILABLE = all(
    path.is_file()
    for path in (
        DEFAULT_ARCHIVE,
        DEFAULT_BODY_LABELS,
        DEFAULT_MAPPER,
        DEFAULT_NSD_DATA_ROOT / "voxels_masks/subj1/cortical_mask_subj01.npy",
        DEFAULT_TENSOR_ROOT / "s1_FFA_t7.pt",
        DEFAULT_TENSOR_ROOT / "s1_EBA_t7.pt",
        DEFAULT_TENSOR_ROOT / "s1_PPA_t7.pt",
    )
)


@unittest.skipUnless(LOCAL_SOURCES_AVAILABLE, "local audited NSD Subject 1 sources unavailable")
class BrainArtifactLocalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selections, _ = load_roi_selections(DEFAULT_NSD_DATA_ROOT, DEFAULT_BODY_LABELS)
        cls.projections, cls.vertex_counts = load_projections(cls.selections, DEFAULT_MAPPER)

    def test_training_compatible_voxel_counts_and_order_hashes(self) -> None:
        expected_counts = {"ffa": 156, "eba": 518, "ppa": 170}
        for roi, selection in self.selections.items():
            with self.subTest(roi=roi):
                self.assertEqual(len(selection.coordinates_zyx), expected_counts[roi])
                digest = hashlib.sha256(
                    selection.coordinates_zyx.astype("<i2").tobytes(order="C")
                ).hexdigest()
                self.assertEqual(digest, EXPECTED_COORDINATE_SHA256[roi])

    def test_linenn_mapping_coverage_is_pinned(self) -> None:
        expected_unmapped = {"ffa": (), "eba": (42, 66, 120), "ppa": ()}
        for roi, projection in self.projections.items():
            with self.subTest(roi=roi):
                self.assertEqual(
                    projection.unmapped_response_indices, expected_unmapped[roi]
                )
                for hemi, mapping in projection.mappings.items():
                    sums = np.bincount(
                        mapping.vertex_indices,
                        weights=mapping.vertex_normalized_weights.astype(np.float64),
                        minlength=self.vertex_counts[hemi],
                    )
                    np.testing.assert_allclose(sums[sums > 0], 1.0, atol=2e-6, rtol=0)

    def test_generated_browser_contract_and_checksums(self) -> None:
        manifest_path = DEFAULT_PUBLIC_OUT / "manifest.json"
        projection_path = DEFAULT_PUBLIC_OUT / "projection.json"
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(projection_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        projection = json.loads(projection_path.read_text(encoding="utf-8"))

        compatibility = {
            "schemaVersion": 1,
            "subjectId": "s1",
            "responseSpace": RESPONSE_SPACE,
            "fLocThreshold": 7,
            "fLocThresholdOperator": ">",
            "responseOrdering": "zyx-c-order",
            "mapper": "linenn",
            "volumeShapeXyz": [83, 104, 81],
            "normalizationPolicy": "per-surface-vertex",
        }
        for key, value in compatibility.items():
            self.assertEqual(manifest[key], value, key)
            self.assertEqual(projection[key], value, key)

        self.assertEqual(manifest["projectionSha256"], sha256_file(projection_path))
        for variant in ("inflated", "pial"):
            surface = manifest["surfaces"][variant]
            path = DEFAULT_PUBLIC_OUT / surface["url"]
            self.assertEqual(path.read_bytes()[:4], b"glTF")
            self.assertEqual(surface["sha256"], sha256_file(path))
            self.assertEqual(surface["hemispheres"]["lh"]["vertexCount"], 227021)
            self.assertEqual(surface["hemispheres"]["rh"]["vertexCount"], 226601)

        full_mapping = DEFAULT_DERIVED_OUT / "subject1_floc_t7_linenn_full.npz"
        self.assertTrue(full_mapping.is_file())
        with np.load(full_mapping, allow_pickle=False) as arrays:
            self.assertFalse(any(arrays[name].dtype.hasobject for name in arrays.files))

    def test_glb_preserves_archive_vertex_and_face_order_exactly(self) -> None:
        try:
            import nibabel.freesurfer as fs
            import trimesh
        except ImportError:
            self.skipTest("nibabel and trimesh are unavailable")

        with tempfile.TemporaryDirectory() as directory:
            surfaces, _ = extract_surface_files(DEFAULT_ARCHIVE, Path(directory))
            for variant in ("inflated", "pial"):
                scene = trimesh.load(
                    DEFAULT_PUBLIC_OUT / f"{variant}.glb", force="scene", process=False
                )
                for hemi in ("lh", "rh"):
                    with self.subTest(variant=variant, hemi=hemi):
                        source_vertices, source_faces = fs.read_geometry(
                            str(surfaces[variant][hemi])
                        )
                        expected_vertices = np.column_stack(
                            (
                                source_vertices[:, 0],
                                source_vertices[:, 2],
                                -source_vertices[:, 1],
                            )
                        ).astype(np.float32)
                        glb_mesh = scene.geometry[hemi]
                        np.testing.assert_array_equal(glb_mesh.vertices, expected_vertices)
                        np.testing.assert_array_equal(glb_mesh.faces, source_faces)


if __name__ == "__main__":
    unittest.main()
