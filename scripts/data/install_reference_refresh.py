#!/usr/bin/env python3
"""Install an explicitly reviewed full reference refresh with recoverable rollback.

The exact parent snapshot is mandatory. All historical raw predictions remain
in place; fresh predictions are appended without filename collisions. Services
must be stopped before invoking this local maintenance command.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

try:
    from . import refresh_reference_coverage as refresh
    from .verify_reference_refresh import verify_refresh_release
    from .verify_coverage_release import child, digest, document
except ImportError:
    import refresh_reference_coverage as refresh  # type: ignore
    from verify_reference_refresh import verify_refresh_release  # type: ignore
    from verify_coverage_release import child, digest, document  # type: ignore


def install_refresh_release(
    stage: Path, spatial_stage: Path, *, parent_snapshot: Path,
    repo: Path = refresh.REPO_ROOT,
    raw_root: Path = refresh.STANDARDIZATION_ROOT / "raw-fp32",
) -> dict:
    report = verify_refresh_release(stage, spatial_stage, repo=repo, baseline=parent_snapshot, raw_root=raw_root)
    standardization_root = repo / "artifacts/precision-standardization"
    backup = standardization_root / f"pre-install-{report['reference_version']}"
    if backup.exists() or backup.is_symlink():
        raise ValueError("Refresh backup already exists; refusing to overwrite retained data")
    current = document(child(repo, "artifacts/reference/manifest.json"))
    reference = document(child(stage, "artifacts/reference/manifest.json"))
    snapshot = document(child(parent_snapshot, "snapshot.json"))
    lineage = reference.get("prediction_standardization", {})
    if (current["reference_version"] != snapshot["reference_version"]
            or lineage.get("policy") != refresh.POLICY
            or lineage.get("parent_reference_version") != current["reference_version"]
            or lineage.get("parent_manifest_sha256") != digest(repo / "artifacts/reference/manifest.json")
            or lineage.get("parent_snapshot_sha256") != digest(parent_snapshot / "snapshot.json")
            or current["supported_models"] != snapshot["models"]
            or current["supported_models"] != reference["supported_models"]
            or lineage.get("regenerated_models") != current["supported_models"]
            or reference["reference_version"] == current["reference_version"]):
        raise ValueError("Refresh does not match the exact reviewed installed parent and full model roster")
    additions, destinations = [], set()
    for model in reference["supported_models"]:
        for roi in reference["rois"]:
            entry = reference["artifacts"][model][roi]
            source = child(raw_root / model, entry["source_file"])
            destination = repo / "data/image_by_model" / source.name
            if destination in destinations or destination.exists() or destination.is_symlink():
                raise ValueError(f"Fresh raw source collides with retained historical data: {destination}")
            if digest(source) != entry["source_sha256"]:
                raise ValueError(f"Fresh raw source changed after verification: {source}")
            destinations.add(destination)
            additions.append((source, destination))
    work = Path(tempfile.mkdtemp(prefix=".reference-refresh-install-", dir=standardization_root))
    moved_originals, installed_roots, installed_sources = [], [], []
    roots = ("public/data", "artifacts/reference")
    try:
        for relative in roots:
            shutil.copytree(stage / relative, work / relative)
        for relative in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
            shutil.copytree(spatial_stage / relative, work / relative, dirs_exist_ok=True)
        actual = {path.relative_to(repo).as_posix()
                  for relative in roots + ("data/image_by_model",)
                  for path in (repo / relative).rglob("*") if path.is_file()}
        if actual != set(snapshot["files"]):
            raise ValueError("Live parent file coverage changed after the explicit snapshot")
        for relative, sha in snapshot["files"].items():
            if digest(child(repo, relative)) != sha:
                raise ValueError(f"Live parent changed during refresh staging: {relative}")
        backup.mkdir()
        for source, destination in additions:
            # Track before copy so even an injected partial-copy failure retains
            # the incomplete new file away from restored live historical data.
            installed_sources.append(destination)
            shutil.copyfile(source, destination)
            if digest(destination) != digest(source):
                raise ValueError(f"Fresh raw copy failed checksum: {destination}")
        for relative in roots:
            old, retained = repo / relative, backup / relative
            retained.parent.mkdir(parents=True, exist_ok=True)
            old.rename(retained)
            moved_originals.append((old, retained))
            (work / relative).rename(old)
            installed_roots.append(old)
        report = {**report, "live_install_performed": True, "backup_root": str(backup),
                  "appended_raw_sources": len(installed_sources), "historical_raw_sources_retired": 0}
        (backup / "install.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return report
    except BaseException:
        for old in reversed(installed_roots):
            failed = backup / "failed-release" / old.relative_to(repo)
            failed.parent.mkdir(parents=True, exist_ok=True)
            old.rename(failed)
        for old, retained in reversed(moved_originals):
            retained.rename(old)
        for path in installed_sources:
            if path.exists():
                retained = backup / "added-raw" / path.name
                retained.parent.mkdir(parents=True, exist_ok=True)
                path.rename(retained)
        raise
    finally:
        # Exact temporary directory owned by this invocation, never an original.
        if work.exists():
            shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--spatial-stage-root", type=Path, required=True)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=refresh.STANDARDIZATION_ROOT / "raw-fp32")
    parser.add_argument("--servers-stopped", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(install_refresh_release(args.stage_root, args.spatial_stage_root,
                                            parent_snapshot=args.parent_snapshot, raw_root=args.raw_root),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
