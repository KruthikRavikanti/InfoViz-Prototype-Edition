#!/usr/bin/env python3
"""Verify a complete staged FP32 refresh against its exact recoverable parent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from . import build_reference_artifacts as builder
    from . import refresh_reference_coverage as refresh
    from .verify_coverage_release import child, digest, document
except ImportError:
    import build_reference_artifacts as builder  # type: ignore
    import refresh_reference_coverage as refresh  # type: ignore
    from verify_coverage_release import child, digest, document  # type: ignore


def validate_refresh_lineage(reference: dict, parent: dict, baseline: Path) -> None:
    standardization = reference.get("prediction_standardization", {})
    models = reference["supported_models"]
    if ("coverage_extension" in reference
            or standardization.get("policy") != refresh.POLICY
            or standardization.get("analysis_input_policy") != refresh.ANALYSIS_INPUT_POLICY
            or standardization.get("parent_reference_version") != parent["reference_version"]
            or standardization.get("parent_manifest_sha256") != digest(baseline / "artifacts/reference/manifest.json")
            or standardization.get("parent_snapshot_sha256") != digest(baseline / "snapshot.json")
            or standardization.get("regenerated_models") != models
            or standardization.get("parent_library_versions") != parent["library_versions"]
            or reference.get("clustering") != parent["clustering"]
            or reference.get("artifact_schema") != parent["artifact_schema"]
            or reference.get("optional_artifacts") != parent["optional_artifacts"]
            or reference.get("checksum") != reference.get("reference_version")
            or reference.get("reference_version") == parent["reference_version"]):
        raise ValueError("Full refresh does not bind its exact parent and unchanged scientific method")
    for name in ("inference_provenance_sha256", "prediction_policies", "model_prediction_policies"):
        if set(standardization.get(name, {})) != set(models):
            raise ValueError(f"Full refresh has incomplete model lineage: {name}")


def load_canonical_source(candidate: builder.SourceCandidate, *, expected_voxel_count: int,
                          expected_means: np.ndarray) -> builder.ValidatedSource:
    """Compare plotted means only after selecting the release's FP32 analysis bits."""
    source = builder.load_and_validate_source(
        candidate, expected_ids=builder.canonical_image_ids(),
        expected_voxel_count=expected_voxel_count, expected_means=None,
    )
    source = refresh.canonicalize_source(source)
    if not np.allclose(source.means, expected_means,
                       atol=builder.MEAN_ABSOLUTE_TOLERANCE, rtol=builder.MEAN_RELATIVE_TOLERANCE):
        raise ValueError(f"Refreshed canonical row means differ from plotted sources: {candidate.model}/{candidate.roi}")
    return source


def verify_refresh_release(
    stage: Path, spatial_stage: Path, *,
    repo: Path = refresh.REPO_ROOT,
    baseline: Path = refresh.STANDARDIZATION_ROOT / "baseline",
    raw_root: Path = refresh.STANDARDIZATION_ROOT / "raw-fp32",
    parent_version: str = refresh.PARENT_REFERENCE_VERSION,
    parent_manifest_sha256: str = refresh.PARENT_MANIFEST_SHA256,
) -> dict:
    parent, snapshot = refresh.validate_parent_snapshot(
        baseline, parent_version=parent_version, parent_manifest_sha256=parent_manifest_sha256,
    )
    artifact_root, data_root = stage / "artifacts/reference", stage / "public/data"
    reference = document(child(artifact_root, "manifest.json"))
    models = refresh.validate_refresh_roster(reference["supported_models"], parent,
                                           baseline / "public/data/murty185_zero_index.json")
    if list(models) != reference["supported_models"]:
        raise ValueError("Refreshed roster must be sorted and unique")
    validate_refresh_lineage(reference, parent, baseline)
    public = document(child(data_root, "reference_manifest.json"))
    if public != builder.public_manifest_from_full(reference):
        raise ValueError("Refreshed public/private manifests disagree")
    report = document(child(stage, "preservation_report.json"))
    if (report.get("policy") != refresh.POLICY or report.get("reference_version") != reference["reference_version"]
            or report.get("parent_reference_version") != parent["reference_version"]
            or report.get("parent_snapshot_sha256") != digest(baseline / "snapshot.json")
            or report.get("regenerated_models") != list(models) or report.get("refitted_models") != list(models)
            or report.get("live_install_performed") is not False):
        raise ValueError("Full refresh preservation report has incorrect lineage")
    rewritten_private = {"manifest.json"} | {
        parent["artifacts"][model][roi]["path"] for model in models for roi in builder.ROIS
    }
    for root_name, changed, report_key in (
        ("artifacts/reference", rewritten_private, "private_artifact_sha256"),
        ("public/data", refresh.REWRITTEN_PUBLIC_PATHS, "public_artifact_sha256"),
    ):
        prefix = root_name + "/"
        expected = {relative[len(prefix):]: sha for relative, sha in snapshot["files"].items()
                    if relative.startswith(prefix) and relative[len(prefix):] not in changed}
        if report.get(report_key) != expected:
            raise ValueError(f"Refresh preservation report omitted or changed shared files: {root_name}")
        for relative, sha in expected.items():
            if digest(child(stage / root_name, relative)) != sha:
                raise ValueError(f"Refresh modified an unrelated parent file: {root_name}/{relative}")

    core_root = repo.parent / "cortex-core"
    sys.path.insert(0, str(core_root))
    sys.path.insert(0, str(repo))
    from cortex.comparison.artifacts import ArtifactRepository
    from cortex.comparison.brain_map_audit import validate_spatial_audit_standardization
    repository = ArtifactRepository(artifact_root)
    if repository.capabilities()["supported_models"] != list(models):
        raise ValueError("Core capabilities differ from the refreshed roster")
    combined_path = child(data_root, "combined_image_means.csv")
    if digest(combined_path) != reference["combined_means"]["sha256"]:
        raise ValueError("Refreshed plotted means failed their checksum")
    combined = builder.load_combined_means(combined_path, models)
    sources = {}
    for model in models:
        for roi in builder.ROIS:
            entry = reference["artifacts"][model][roi]
            source_path = child(raw_root / model, entry["source_file"])
            if f"data/image_by_model/{entry['source_file']}" in snapshot["files"]:
                raise ValueError("Refreshed source collides with a retained parent raw prediction")
            if digest(source_path) != entry["source_sha256"]:
                raise ValueError(f"Refreshed raw prediction changed: {model}/{roi}")
            source = load_canonical_source(
                builder.SourceCandidate(model, roi, entry["source_timestamp"], source_path),
                expected_voxel_count=builder.VOXEL_COUNTS[roi],
                expected_means=combined[f"{model}_{roi}"].to_numpy(dtype=np.float64),
            )
            artifact = repository.load(model, roi)
            if (not np.array_equal(artifact.reference_voxels, source.voxels.astype(np.float32))
                    or not np.array_equal(artifact.reference_means, source.means)):
                raise ValueError(f"Refreshed NPZ raw matrix/means differ from plotted sources: {model}/{roi}")
            sources[(model, roi)] = source
    proofs = refresh.validate_inference_proofs(models=models, raw_root=raw_root, sources=sources,
                                               core_root=core_root, image_root=repo / "public/images")
    lineage = reference["prediction_standardization"]
    for model in models:
        if (digest(raw_root / model / "provenance.json") != lineage["inference_provenance_sha256"][model]
                or refresh.expansion.precision_policy_from_provenance(model, proofs[model]) != lineage["prediction_policies"][model]
                or refresh.expansion.prediction_policy_from_provenance(model, proofs[model]) != lineage["model_prediction_policies"][model]):
            raise ValueError(f"Refreshed inference provenance/policy changed: {model}")
        for roi in builder.ROIS:
            key = f"{model}/{roi}"
            selection = reference["source_selection"][key]
            entry = reference["artifacts"][model][roi]
            if (selection.get("selected") != entry["source_file"]
                    or selection.get("selected_timestamp") != entry["source_timestamp"]
                    or selection.get("raw_relative_path") != f"{model}/{entry['source_file']}"
                    or selection.get("inference_provenance_sha256") != lineage["inference_provenance_sha256"][model]
                    or selection.get("candidate_count") != 1 or selection.get("rejected_newer_candidates") != []):
                raise ValueError(f"Refreshed source selection is not its exact fresh inference: {key}")
    contracts = builder.load_static_voxel_cluster_contracts(
        sources, kmeans_dir=data_root / "voxel_clusters", hierarchical_dir=data_root / "voxel_clusters_hierarchical_corr",
    )
    for (model, roi), contract in contracts.items():
        artifact = repository.load(model, roi)
        if (not np.array_equal(artifact.kmeans_labels, contract.kmeans_labels)
                or not np.array_equal(artifact.hierarchical_labels, contract.hierarchical_labels)
                or not np.allclose(artifact.plot_reference_coordinates, contract.plot_coordinates, rtol=0, atol=1e-9)):
            raise ValueError(f"Refreshed plotted clusters/coordinates differ from Core: {model}/{roi}")
    visual = reference["optional_artifacts"]["visual"]
    static_sha = builder.sha256_json({
        "voxel": {f"{model}/{roi}": {"kmeans": contract.kmeans_sha256, "hierarchical": contract.hierarchical_sha256}
                  for (model, roi), contract in sorted(contracts.items())},
        "visual": {"kmeans": visual["static_kmeans_sha256"], "hierarchical": visual["static_hierarchical_sha256"]},
    })
    identity = builder._reference_identity(
        models=models, image_ids=builder.canonical_image_ids(), combined_means_sha256=digest(combined_path),
        selected=sources, visual_features_sha256=visual["source_sha256"],
        visual_images_sha256=visual["source_images_sha256"], static_clusters_sha256=static_sha,
    )
    identity["prediction_standardization"] = lineage
    if builder.sha256_json(identity) != reference["reference_version"]:
        raise ValueError("Refreshed reference identity does not reproduce its full version hash")
    brain_root = spatial_stage / "public/data/brain/reference"
    atlas_root = spatial_stage / "public/data/voxel_selectivity_atlas"
    brain, atlas = document(child(brain_root, "manifest.json")), document(child(atlas_root, "manifest.json"))
    audit = document(child(brain_root, "spatial-audit.json"))
    parent_brain = document(child(baseline, "public/data/brain/reference/manifest.json"))
    parent_atlas = document(child(baseline, "public/data/voxel_selectivity_atlas/manifest.json"))
    for old, new, kind in ((parent_brain, brain, "brain"), (parent_atlas, atlas, "atlas")):
        for key, value in old.items():
            if key not in {"models", "referenceVersion"} and new.get(key) != value:
                raise ValueError(f"Full refresh changed the existing {kind} scientific method/identity: {key}")
    if (brain["referenceVersion"] != reference["reference_version"] or atlas["referenceVersion"] != reference["reference_version"]
            or set(brain["models"]) != set(models) or set(atlas["models"]) != set(models)
            or atlas["supportedModels"] != list(models)
            or set(validate_spatial_audit_standardization(audit, reference_version=reference["reference_version"])) != set(models)):
        raise ValueError("Refreshed reference, spatial manifests and audit disagree")
    if (audit.get("parent_reference_version") != parent["reference_version"]
            or audit.get("parent_manifest_sha256") != lineage["parent_manifest_sha256"]
            or audit.get("parent_snapshot_sha256") != lineage["parent_snapshot_sha256"]
            or audit.get("parent_spatial_audit_sha256") != digest(baseline / "public/data/brain/reference/spatial-audit.json")
            or audit.get("checkpoint_sha256_by_model") != document(baseline / "public/data/brain/reference/spatial-audit.json")["checkpoint_sha256_by_model"]
            or audit.get("inference_provenance_sha256_by_model") != lineage["inference_provenance_sha256"]
            or audit.get("model_prediction_policies") != lineage["model_prediction_policies"]):
        raise ValueError("Refreshed spatial audit does not bind the same full prediction lineage")
    for model, entry in brain["models"].items():
        path = child(brain_root, entry["url"])
        if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Refreshed brain payload failed its checksum: {model}")
        matrix = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(builder.REFERENCE_COUNT, brain["rowWidth"])
        for roi, layout in brain["roiLayout"].items():
            offset, count = layout["offset"], layout["count"]
            if not np.array_equal(matrix[:, offset:offset + count], sources[(model, roi)].voxels[:, :count].astype(np.float32)):
                raise ValueError(f"Refreshed brain payload is not its raw Subject 1 source: {model}/{roi}")
    for model, entry in atlas["models"].items():
        for kind in ("points", "summary"):
            path = child(atlas_root, entry[f"{kind}Url"])
            if digest(path) != entry[f"{kind}Sha256"] or path.stat().st_size != entry[f"{kind}Bytes"]:
                raise ValueError(f"Refreshed atlas payload failed its checksum: {model}/{kind}")
    for root_name, old_manifest in (("brain/reference", parent_brain), ("voxel_selectivity_atlas", parent_atlas)):
        rewritten = {"manifest.json"}
        if root_name == "brain/reference":
            rewritten.add("spatial-audit.json")
            rewritten.update(entry["url"] for entry in old_manifest["models"].values())
        else:
            rewritten.update(entry[f"{kind}Url"] for entry in old_manifest["models"].values() for kind in ("points", "summary"))
        prefix = f"public/data/{root_name}/"
        for relative, sha in snapshot["files"].items():
            if relative.startswith(prefix) and relative[len(prefix):] not in rewritten:
                if digest(child(spatial_stage, relative)) != sha:
                    raise ValueError(f"Refresh changed an unrelated spatial parent file: {relative}")
    return {
        "policy": refresh.POLICY, "reference_version": reference["reference_version"],
        "parent_reference_version": parent["reference_version"], "parent_snapshot_sha256": lineage["parent_snapshot_sha256"],
        "model_count": len(models), "regenerated_models": list(models), "roi_count": len(builder.ROIS),
        "reference_count": builder.REFERENCE_COUNT, "shared_artifacts_preserved": True,
        "brain_manifest_sha256": digest(brain_root / "manifest.json"), "atlas_manifest_sha256": digest(atlas_root / "manifest.json"),
        "spatial_audit_sha256": digest(brain_root / "spatial-audit.json"), "live_install_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--spatial-stage-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=refresh.STANDARDIZATION_ROOT / "baseline")
    parser.add_argument("--raw-root", type=Path, default=refresh.STANDARDIZATION_ROOT / "raw-fp32")
    args = parser.parse_args()
    import json
    print(json.dumps(verify_refresh_release(args.stage_root, args.spatial_stage_root,
                                           baseline=args.baseline_root, raw_root=args.raw_root), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
