#!/usr/bin/env python3
"""Install a verified append-only bundle during an explicit local maintenance.

Running services must be stopped first. Original live directories are moved
into a version-specific backup; no originals are deleted. A failed installation
restores both original directories and moves newly added sources to the backup.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

try:
    from .verify_coverage_release import BASELINE, REPO_ROOT, child, digest, document, verify_release
except ImportError:
    from verify_coverage_release import BASELINE, REPO_ROOT, child, digest, document, verify_release


def install_release(stage: Path, spatial_stage: Path, *, repo: Path = REPO_ROOT,
                    baseline: Path = BASELINE, raw_root: Path | None = None,
                    replace_snapshot: Path | None = None) -> dict:
    raw_root = raw_root or repo / "artifacts/coverage-expansion/raw"
    report = verify_release(stage, spatial_stage, repo=repo, baseline=baseline, raw_root=raw_root)
    coverage_root = repo / "artifacts/coverage-expansion"
    version = report["reference_version"]
    backup = coverage_root / f"pre-install-{version}"
    if backup.exists():
        raise ValueError(f"Install backup already exists; refusing replacement: {backup}")
    current = document(repo / "artifacts/reference/manifest.json")
    original = document(baseline / "artifacts/reference/manifest.json")
    reference = document(stage / "artifacts/reference/manifest.json")
    snapshot = document((replace_snapshot or baseline) / "snapshot.json")
    retired_sources = []
    if replace_snapshot is None:
        if current["reference_version"] != original["reference_version"]:
            raise ValueError("Live reference is no longer the reviewed baseline")
    else:
        # Correcting a reviewed candidate is not permission to replace arbitrary
        # data. Require the exact current snapshot, same roster, and original
        # immutable baseline lineage on both candidates.
        if (current["reference_version"] != snapshot["reference_version"]
                or current["reference_version"] == reference["reference_version"]
                or current["supported_models"] != snapshot["models"]
                or current["supported_models"] != reference["supported_models"]
                or current.get("coverage_extension", {}).get("parent_reference_version") != original["reference_version"]
                or reference.get("coverage_extension", {}).get("parent_reference_version") != original["reference_version"]):
            raise ValueError("Replacement does not match the reviewed installed candidate and baseline lineage")
        baseline_files = set(document(baseline / "snapshot.json")["files"])
        old_destinations = set()
        for model in report["added_models"]:
            for roi in reference["rois"]:
                entry = current["artifacts"][model][roi]
                source = child(repo / "data/image_by_model", entry["source_file"])
                relative = source.relative_to(repo).as_posix()
                if (source in old_destinations or relative in baseline_files
                        or relative not in snapshot["files"]
                        or digest(source) != entry["source_sha256"]):
                    raise ValueError(f"Unsafe or changed prior candidate source: {relative}")
                old_destinations.add(source)
                retired_sources.append(source)
    additions = []
    destinations = set()
    for model in report["added_models"]:
        for roi in reference["rois"]:
            entry = reference["artifacts"][model][roi]
            source = child(raw_root / model, entry["source_file"])
            destination = repo / "data/image_by_model" / source.name
            if destination in destinations or destination.exists() or destination.is_symlink():
                raise ValueError(f"New raw source collides with existing data: {destination}")
            destinations.add(destination)
            additions.append((source, destination))
    work = Path(tempfile.mkdtemp(prefix=".release-install-", dir=coverage_root))
    moved_originals, installed_roots, installed_sources, retired_originals = [], [], [], []
    roots = ("public/data", "artifacts/reference")
    try:
        for relative in roots:
            shutil.copytree(stage / relative, work / relative)
        # Overwrite only our own freshly copied, checksum-verified spatial copy.
        for relative in ("public/data/brain/reference", "public/data/voxel_selectivity_atlas"):
            shutil.copytree(spatial_stage / relative, work / relative, dirs_exist_ok=True)
        # Refuse a concurrent user edit, including a newly added untracked file.
        live_files = {path.relative_to(repo).as_posix()
                      for relative in roots + ("data/image_by_model",)
                      for path in (repo / relative).rglob("*") if path.is_file()}
        if live_files != set(snapshot["files"]):
            raise ValueError("Live baseline file coverage changed after the snapshot")
        for relative, expected in snapshot["files"].items():
            if digest(child(repo, relative)) != expected:
                raise ValueError(f"Live baseline changed during staging: {relative}")
        backup.mkdir()
        for source, destination in additions:
            installed_sources.append(destination)
            shutil.copyfile(source, destination)
            if digest(source) != digest(destination):
                raise ValueError(f"New raw source copy failed: {source.name}")
        for source in retired_sources:
            retained = backup / "prior-added-raw" / source.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            source.rename(retained)
            retired_originals.append((source, retained))
        for relative in roots:
            old = repo / relative
            retained = backup / relative
            retained.parent.mkdir(parents=True, exist_ok=True)
            old.rename(retained)
            moved_originals.append((old, retained))
            (work / relative).rename(old)
            installed_roots.append(old)
        report = {**report, "live_install_performed": True, "backup_root": str(backup),
                  "replaced_reference_version": current["reference_version"] if replace_snapshot else None,
                  "retained_prior_raw_sources": len(retired_originals)}
        (backup / "install.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return report
    except BaseException:
        # Preserve failed release data too, and restore exact pre-install state.
        for old in reversed(installed_roots):
            failed = backup / "failed-release" / old.relative_to(repo)
            failed.parent.mkdir(parents=True, exist_ok=True)
            old.rename(failed)
        for old, retained in reversed(moved_originals):
            retained.rename(old)
        for old, retained in reversed(retired_originals):
            retained.rename(old)
        for source in installed_sources:
            if not source.exists():
                continue
            retained = backup / "added-raw" / source.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            source.rename(retained)
        raise
    finally:
        # This exact directory was created by this invocation; originals never land here.
        if work.exists():
            shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--spatial-stage-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--replace-installed-snapshot", type=Path,
                        help="Explicit byte-verified snapshot of a same-roster candidate to correct; originals are retained")
    parser.add_argument("--servers-stopped", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(install_release(args.stage_root, args.spatial_stage_root,
                                    raw_root=args.raw_root,
                                    replace_snapshot=args.replace_installed_snapshot), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
