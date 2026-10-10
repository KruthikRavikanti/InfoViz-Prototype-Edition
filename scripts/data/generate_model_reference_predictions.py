#!/usr/bin/env python3
"""Stage genuine NSD-1000 predictions for missing Cortex-Compass models.

Nothing in the live reference bundle is replaced. Each completed model is an
atomically published directory of three raw CSVs and a provenance record; only
verified complete triads may be resumed. Imports of the inference stack are
deferred so --preflight can inspect checkpoints without loading any backbone.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROIS = ("ffa", "ppa", "eba")
DATASET = "nsd_1000"
IMAGE_IDS = tuple(f"image_{i:03d}.png" for i in range(1, 186))
SUBJECT_ORDER = ("s1", "s2", "s5", "s7")
SUBJECT_VOXEL_COUNTS = {
    "s1": {"ffa": 156, "ppa": 170, "eba": 518},
    "s2": {"ffa": 218, "ppa": 224, "eba": 526},
    "s5": {"ffa": 271, "ppa": 295, "eba": 682},
    "s7": {"ffa": 71, "ppa": 64, "eba": 319},
}
VOXEL_COUNTS = {roi: sum(SUBJECT_VOXEL_COUNTS[s][roi] for s in SUBJECT_ORDER) for roi in ROIS}
PREPROCESSING = {
    "image_mode": "RGB",
    "square_padding": True,
    "pad_color": 0,
    "padding_placement": "center_floor_offset",
    "backbone_transforms": "unmodified_cortex_model_transforms",
    "inference_call": "ReadoutModel.inference(return_logits=True, do_square_padding=True)",
    "voxel_normalization": "none",
    "tail_policy": "none",
}
NUMERIC_POLICY = {
    "policy": "fp32_no_tf32_v1",
    "output_dtype": "float32",
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_autocast_enabled": False,
    "cpu_autocast_enabled": False,
}


class GenerationValidationError(ValueError):
    pass


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_voxel_columns(roi: str) -> list[str]:
    if roi not in ROIS:
        raise GenerationValidationError(f"Unsupported ROI: {roi}")
    return [f"{subject}_{i}" for subject in SUBJECT_ORDER for i in range(SUBJECT_VOXEL_COUNTS[subject][roi])]


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise GenerationValidationError(f"Cannot read literal {name} from {path}")


def model_inventory(core_root: Path, checkpoint_root: Path, scores_path: Path, reference_manifest: Path) -> dict[str, dict]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scored = set().union(*(set(values) for values in scores.values())) - {"ceiling"}
    layers = literal_assignment(core_root / "cortex/constants/best_readout_layers.py", "best_readout_layers")[DATASET]
    tree = ast.parse((core_root / "cortex/models/__init__.py").read_text(encoding="utf-8"))
    registry = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "backbone_models" for t in n.targets))
    registered = {ast.literal_eval(value)["name"] for value in registry.values}
    present = set(json.loads(reference_manifest.read_text(encoding="utf-8"))["supported_models"])
    if not present.issubset(scored):
        raise GenerationValidationError("The active reference roster contains an unscored model.")
    result = {}
    for model in sorted(scored):
        if not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", model) or model not in registered:
            raise GenerationValidationError(f"Scored model is not a safe registered backbone: {model}")
        triad = {}
        for roi in ROIS:
            layer = layers.get(roi, {}).get(model)
            path = checkpoint_root / DATASET / f"{model}_{DATASET}_all_{roi}_{layer}.model" if layer else None
            triad[roi] = {
                "readout_layer": layer,
                "checkpoint_path": str(path) if path else None,
                "readable": bool(path and path.is_file() and os.access(path, os.R_OK)),
                "output_width": VOXEL_COUNTS[roi],
            }
        result[model] = {"existing_reference": model in present, "rois": triad}
    return result


def select_models(inventory: dict[str, dict], models: list[str] | None, all_missing: bool) -> list[str]:
    if bool(models) == all_missing:
        raise GenerationValidationError("Choose exactly one of --models or --all-missing.")
    selected = sorted(set(models)) if models else [m for m, entry in inventory.items() if not entry["existing_reference"]]
    unknown = set(selected) - set(inventory)
    if unknown:
        raise GenerationValidationError("Models are not on the scored roster: " + ", ".join(sorted(unknown)))
    return selected


def image_provenance(image_root: Path) -> list[dict]:
    result = []
    for image_id in IMAGE_IDS:
        path = image_root / image_id
        if not path.is_file():
            raise GenerationValidationError(f"Missing canonical reference image: {image_id}")
        result.append({"id": image_id, "sha256": sha256_file(path)})
    return result


def validate_predictions(matrix: Any, roi: str) -> Any:
    import numpy as np
    matrix = np.asarray(matrix)
    if matrix.shape != (len(IMAGE_IDS), VOXEL_COUNTS[roi]) or matrix.dtype != np.float32:
        raise GenerationValidationError(f"{roi}: inference must return finite float32 ({len(IMAGE_IDS)}, {VOXEL_COUNTS[roi]}) raw voxels, got {matrix.shape}/{matrix.dtype}.")
    if not np.isfinite(matrix).all():
        raise GenerationValidationError(f"{roi}: non-finite predictions.")
    return matrix


def write_prediction_csv(path: Path, roi: str, matrix: Any) -> None:
    matrix = validate_predictions(matrix, roi)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image", *ordered_voxel_columns(roi)])
        # Nine significant digits round-trip every finite float32 prediction.
        for image_id, row in zip(IMAGE_IDS, matrix):
            writer.writerow([image_id, *(format(float(v), ".9g") for v in row)])


def read_verified_csv(path: Path, roi: str) -> Any:
    import numpy as np
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["image", *ordered_voxel_columns(roi)]:
            raise GenerationValidationError(f"{roi}: exported voxel header is not the complete canonical subject order.")
        rows = list(reader)
    if tuple(row[0] for row in rows if row) != IMAGE_IDS:
        raise GenerationValidationError(f"{roi}: exported reference image order changed.")
    try:
        matrix = np.asarray([row[1:] for row in rows], dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise GenerationValidationError(f"{roi}: invalid numeric CSV.") from exc
    return validate_predictions(matrix, roi)


def verify_completed_triad(model_root: Path, model: str, inventory_entry: dict, images: list[dict]) -> dict:
    provenance = json.loads((model_root / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("schema_version") != 1 or provenance.get("model") != model or provenance.get("dataset") != DATASET
            or provenance.get("image_ids") != list(IMAGE_IDS) or provenance.get("images") != images
            or provenance.get("image_set_sha256") != canonical_json_sha256(images)
            or provenance.get("subject_order") != list(SUBJECT_ORDER)
            or provenance.get("subject_voxel_counts") != SUBJECT_VOXEL_COUNTS
            or provenance.get("voxel_counts") != VOXEL_COUNTS or provenance.get("preprocessing") != PREPROCESSING
            or set(provenance.get("rois", {})) != set(ROIS)):
        raise GenerationValidationError(f"{model}: incomplete or incompatible existing provenance; refusing to overwrite.")
    for roi in ROIS:
        entry = provenance["rois"][roi]
        name = entry.get("path", "")
        if Path(name).name != name or not re.fullmatch(rf"murtylab_{re.escape(model)}_{DATASET}_{roi}_\d{{17}}Z\.csv", name):
            raise GenerationValidationError(f"{model}/{roi}: unsafe or invalid staged filename.")
        path = model_root / name
        checkpoint = Path(inventory_entry["rois"][roi]["checkpoint_path"])
        columns = ordered_voxel_columns(roi)
        if (entry.get("sha256") != sha256_file(path) or entry.get("checkpoint_sha256") != sha256_file(checkpoint)
                or entry.get("checkpoint_path") != str(checkpoint)
                or entry.get("readout_layer") != inventory_entry["rois"][roi]["readout_layer"]
                or entry.get("output_shape") != [len(IMAGE_IDS), VOXEL_COUNTS[roi]]
                or entry.get("voxel_columns") != columns or entry.get("voxel_order_sha256") != canonical_json_sha256(columns)):
            raise GenerationValidationError(f"{model}/{roi}: existing CSV/checkpoint provenance does not verify.")
        read_verified_csv(path, roi)
    return provenance


def verify_precision_provenance(provenance: dict, core_root: Path) -> None:
    policy = provenance.get("numeric_policy")
    if (not isinstance(policy, dict) or policy != NUMERIC_POLICY
            or any(type(policy.get(key)) is not type(value) for key, value in NUMERIC_POLICY.items())):
        raise GenerationValidationError("Missing or incompatible observed FP32 numeric policy; refusing resume.")
    helper = core_root / "cortex/utils/inference_precision.py"
    helper_hash = sha256_file(helper)
    if (provenance.get("precision_helper_sha256") != helper_hash
            or provenance.get("core_inference_source_sha256", {}).get("cortex/utils/inference_precision.py") != helper_hash):
        raise GenerationValidationError("FP32 precision-helper source identity does not verify; refusing resume.")


def prepare_backbone_overlay(source_root: Path, target_root: Path) -> dict:
    """Expose readable shared assets through immutable file links, not dir links.

    Real local directories ensure constructor lockfiles/downloads cannot land in
    the shared cache. Mutable tiny refs/trust metadata are copied locally; model
    weight bytes are linked and never duplicated. Permission-denied assets are
    recorded, not bypassed or chmodded.
    """
    source_root, target_root = source_root.resolve(), target_root.resolve()
    allowed = (REPO_ROOT / "artifacts/coverage-expansion").resolve()
    if not target_root.is_relative_to(allowed) or target_root == allowed or source_root == target_root:
        raise GenerationValidationError("Backbone overlay must be a project-local coverage-expansion subdirectory.")
    if not source_root.is_dir():
        raise GenerationValidationError("Shared backbone cache source does not exist.")
    target_root.mkdir(parents=True, exist_ok=True)
    linked, copied, existing, excluded = [], [], [], []

    def walk_error(error):
        excluded.append({"path": str(error.filename), "reason": "unreadable_directory"})

    for directory, dirs, files in os.walk(source_root, onerror=walk_error, followlinks=False):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".locks")]
        relative_dir = Path(directory).relative_to(source_root)
        local_dir = target_root / relative_dir
        if local_dir.is_symlink():
            raise GenerationValidationError("Overlay directories must not link into the shared cache.")
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            source = Path(directory) / name
            relative = relative_dir / name
            destination = target_root / relative
            if not source.is_file() or not os.access(source, os.R_OK):
                excluded.append({"path": str(source), "reason": "unreadable_file"})
                continue
            if destination.exists() or destination.is_symlink():
                existing.append(str(relative))
                continue
            if "refs" in relative.parts or name == "trusted_list":
                if source.stat().st_size > 1024 * 1024:
                    raise GenerationValidationError("Unexpectedly large mutable cache metadata.")
                destination.write_bytes(source.read_bytes())
                copied.append(str(relative))
            else:
                destination.symlink_to(source.resolve())
                linked.append(str(relative))
    manifest = {"schema_version": 1, "source_root": str(source_root), "overlay_root": str(target_root),
                "policy": "real_directories_readable_weight_file_links_local_mutable_metadata",
                "linked": linked, "copied_metadata": copied, "preserved_existing": existing,
                "excluded": excluded, "created_at": datetime.now(timezone.utc).isoformat()}
    (target_root / "overlay_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


@contextmanager
def shared_research_write_guard():
    """Prevent writes through cache file symlinks into either shared volume."""
    import builtins
    import io
    originals = (builtins.open, io.open, os.open)

    def check(path, writing):
        if writing and not isinstance(path, int):
            resolved = Path(path).resolve()
            if any(resolved.is_relative_to(root) for root in (Path("/research"), Path("/research2"))):
                raise GenerationValidationError("Refusing a write to the shared research cache; use the project-local overlay.")

    def guarded_builtin(file, mode="r", *args, **kwargs):
        check(file, any(flag in mode for flag in "wax+"))
        return originals[0](file, mode, *args, **kwargs)

    def guarded_io(file, mode="r", *args, **kwargs):
        check(file, any(flag in mode for flag in "wax+"))
        return originals[1](file, mode, *args, **kwargs)

    def guarded_os(path, flags, *args, **kwargs):
        check(path, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        return originals[2](path, flags, *args, **kwargs)

    builtins.open, io.open, os.open = guarded_builtin, guarded_io, guarded_os
    try:
        yield
    finally:
        builtins.open, io.open, os.open = originals


@contextmanager
def offline_network_guard(enabled: bool):
    """Deny missing-weight downloads, including legacy urllib/request loaders."""
    if not enabled:
        yield
        return
    import urllib.request
    from urllib.error import URLError
    import requests
    import socket
    import torch.hub
    from huggingface_hub.errors import OfflineModeIsEnabled
    originals = (urllib.request.urlopen, urllib.request.urlretrieve, requests.sessions.Session.request, os.system,
                 torch.hub.urlopen, socket.socket.connect)

    def safe_url(value):
        from urllib.parse import urlsplit
        if hasattr(value, "get_full_url"):
            value = value.get_full_url()
        if not isinstance(value, str):
            return ""
        parsed = urlsplit(value)
        # Never include credentials, queries, or fragments in failure records.
        return f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}" if parsed.hostname else ""

    def denied(*args, **kwargs):
        url = safe_url(args[0]) if args else ""
        raise GenerationValidationError("Network request blocked: offline generation requires cached backbone assets." + (f" URL: {url}" if url else ""))

    def offline_hub_request(*args, **kwargs):
        # Hugging Face catches this precise exception and falls back to a
        # verified local cache. A generic ValueError would break that fallback.
        url = safe_url(args[2] if len(args) > 2 else kwargs.get("url"))
        raise OfflineModeIsEnabled("Offline reference generation requires cached backbone assets." + (f" URL: {url}" if url else ""))

    def offline_torch_hub_urlopen(*args, **kwargs):
        # Torch Hub probes a repository's default branch before looking for its
        # cached checkout. Its documented offline fallback catches URLError;
        # preserve that fallback while still denying every network request.
        url = safe_url(args[0]) if args else ""
        raise URLError("Network request blocked: offline generation requires cached backbone assets." + (f" URL: {url}" if url else ""))

    def guarded_system(command):
        if re.search(r"\b(?:wget|curl|gdown)\b|\bgit\s+clone\b", command):
            return denied()
        return originals[3](command)

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            return denied()
        return originals[5](sock, address)

    urllib.request.urlopen = denied
    urllib.request.urlretrieve = denied
    requests.sessions.Session.request = offline_hub_request
    os.system = guarded_system
    # Torch Hub imports urlopen by value; replacing urllib's attribute alone
    # does not stop torchvision checkpoint downloads through that private alias.
    torch.hub.urlopen = offline_torch_hub_urlopen
    socket.socket.connect = guarded_connect
    try:
        yield
    finally:
        (urllib.request.urlopen, urllib.request.urlretrieve, requests.sessions.Session.request, os.system,
         torch.hub.urlopen, socket.socket.connect) = originals


def backbone_state_sha256(backbone: Any) -> str:
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        descriptor = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        # uint8 also supports bfloat16, whose direct NumPy conversion is absent.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_parallel_execution(backbone: Any, device: str) -> None:
    """Readout hooks are not replica-safe; require one visible CUDA device."""
    import torch
    if (device.startswith("cuda") and torch.cuda.device_count() > 1
            and any(isinstance(module, torch.nn.DataParallel) for module in backbone.modules())):
        raise GenerationValidationError(
            "DataParallel readout hooks require one visible GPU. Restart with "
            "CUDA_VISIBLE_DEVICES=0 and --device cuda:0 before importing Torch; "
            "weights, wrappers, and noise modes must remain unchanged."
        )


def library_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "torchvision", "transformers", "timm", "numpy", "pillow", "einops", "dreamsim", "dinov2", "cornet", "vonenet"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def core_source_identity(core_root: Path) -> dict:
    import subprocess
    files = sorted((core_root / "cortex/models").rglob("*.py"))
    files.extend(core_root / file for file in (
        "cortex/pretrained.py", "cortex/constants/best_readout_layers.py", "cortex/datasets/nsd/rois.py",
        "cortex/datasets/nsd/dataset.py", "cortex/utils/seed.py", "cortex/utils/padding.py",
        "cortex/utils/hook.py", "cortex/utils/getting_modules.py",
        "cortex/inference_service.py", "cortex/utils/inference_precision.py", "cortex/utils/backbone_cache.py",
    ))
    result = subprocess.run(["git", "status", "--porcelain"], cwd=core_root, capture_output=True, text=True, check=False)
    return {"core_git_dirty": bool(result.stdout.strip()) if result.returncode == 0 else None,
            "core_inference_source_sha256": {str(path.relative_to(core_root)): sha256_file(path) for path in files}}


def generate_model(model: str, *, inventory_entry: dict, core_root: Path, image_root: Path, backbone_root: Path,
                   output_root: Path, images_record: list[dict], device: str, batch_size: int, seed: int, offline: bool) -> dict:
    import gc
    import numpy as np
    import torch
    from PIL import Image
    from cortex.models import load_backbone_by_name
    from cortex.pretrained import from_pretrained
    from cortex.utils.seed import seed_everything
    from cortex.datasets.nsd.dataset import valid_subject_ids
    from cortex.datasets.nsd.rois import roi_indices
    from cortex.utils.inference_precision import FULL_FP32_NUMERIC_POLICY, observed_numeric_policy, scoped_inference_precision
    from cortex.utils.backbone_cache import scoped_backbone_cache

    if NUMERIC_POLICY != FULL_FP32_NUMERIC_POLICY:
        raise GenerationValidationError("Generator and Core numeric policy contracts disagree.")

    if list(valid_subject_ids) != list(SUBJECT_ORDER) or list(roi_indices) != list(SUBJECT_ORDER):
        raise GenerationValidationError("Core NSD subject ordering differs from the canonical output contract.")
    for subject in SUBJECT_ORDER:
        for roi in ROIS:
            index = roi_indices[subject][roi]
            if index["end"] - index["start"] != SUBJECT_VOXEL_COUNTS[subject][roi]:
                raise GenerationValidationError("Core NSD ROI sizes differ from the canonical output contract.")
    destination = output_root / model
    if destination.exists():
        completed = verify_completed_triad(destination, model, inventory_entry, images_record)
        verify_precision_provenance(completed, core_root)
        return completed
    if not all(entry["readable"] for entry in inventory_entry["rois"].values()):
        raise GenerationValidationError(f"{model}: missing/unreadable configured readout checkpoint.")
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{model}.tmp-", dir=output_root))
    backbone = None
    readout = None
    logits = None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17] + "Z"
    try:
        pil_images = []
        for image_id in IMAGE_IDS:
            with Image.open(image_root / image_id) as image:
                pil_images.append(image.convert("RGB").copy())
        seed_everything(seed)
        with (offline_network_guard(offline), shared_research_write_guard(), scoped_backbone_cache(backbone_root),
              scoped_inference_precision(policy=NUMERIC_POLICY["policy"]) as numeric_policy):
            backbone = load_backbone_by_name(model, pretrained=True, download_root=str(backbone_root)).eval()
            validate_parallel_execution(backbone, device)
            state_hash = backbone_state_sha256(backbone)
            noise_modes = sorted({str(module.noise_mode) for module in backbone.modules() if getattr(module, "noise_mode", None) is not None})
            provenance = {
                "schema_version": 1, "model": model, "dataset": DATASET, "reference_count": len(IMAGE_IDS),
                "image_ids": list(IMAGE_IDS), "images": images_record, "image_set_sha256": canonical_json_sha256(images_record),
                "subject_order": list(SUBJECT_ORDER), "subject_voxel_counts": SUBJECT_VOXEL_COUNTS,
                "voxel_counts": VOXEL_COUNTS, "preprocessing": PREPROCESSING, "seed": seed,
                "seed_policy": "constructor_and_each_roi_inference", "batch_size": batch_size,
                "device": device, "offline": offline, "generated_at": datetime.now(timezone.utc).isoformat(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "backbone_cache_root": str(backbone_root),
                "core_commit": core_commit(core_root), "library_versions": library_versions(),
                "generator_sha256": GENERATOR_SHA256, **core_source_identity(core_root),
                "numeric_policy": numeric_policy,
                "precision_helper_sha256": sha256_file(core_root / "cortex/utils/inference_precision.py"),
                "backbone_class": f"{type(backbone).__module__}.{type(backbone).__qualname__}",
                "backbone_state_sha256": state_hash, "noise_modes": noise_modes,
                "deterministic_per_stimulus": not bool(noise_modes),
                "response_policy": "seeded_stochastic_sampling" if noise_modes else "evaluation_mode",
                "rois": {},
            }
            for roi in ROIS:
                entry = inventory_entry["rois"][roi]
                checkpoint = Path(entry["checkpoint_path"])
                checkpoint_hash = sha256_file(checkpoint)
                readout = from_pretrained(backbone_name=model, dataset_name=DATASET, subject_id="all", roi=roi,
                                          checkpoint_folder=str(checkpoint.parent), device=device,
                                          readout_layer=entry["readout_layer"], backbone=backbone,
                                          backbone_download_root=str(backbone_root)).eval()
                mapper = readout.readout_model
                if (getattr(mapper, "roi", None) != roi or getattr(mapper, "output_size", getattr(mapper, "out_features", None)) != VOXEL_COUNTS[roi]
                        or readout.dataset_name != DATASET or readout.checkpoint_sha256 != checkpoint_hash):
                    raise GenerationValidationError(f"{model}/{roi}: checkpoint identity, stored ROI, or readout output width does not match.")
                seed_everything(seed)
                if observed_numeric_policy(NUMERIC_POLICY["policy"]) != NUMERIC_POLICY:
                    raise GenerationValidationError(f"{model}/{roi}: constructor changed the observed FP32 policy.")
                logits = readout.inference(images=pil_images, return_logits=True, do_square_padding=True, batch_size=batch_size)
                if observed_numeric_policy(NUMERIC_POLICY["policy"]) != NUMERIC_POLICY:
                    raise GenerationValidationError(f"{model}/{roi}: inference changed the observed FP32 policy.")
                matrix = validate_predictions(logits.detach().cpu().numpy(), roi)
                filename = f"murtylab_{model}_{DATASET}_{roi}_{timestamp}.csv"
                csv_path = temp_root / filename
                write_prediction_csv(csv_path, roi, matrix)
                if not np.array_equal(read_verified_csv(csv_path, roi), matrix):
                    raise GenerationValidationError(f"{model}/{roi}: CSV did not round-trip the raw float32 predictions.")
                columns = ordered_voxel_columns(roi)
                provenance["rois"][roi] = {
                    "path": filename, "sha256": sha256_file(csv_path), "checkpoint_path": str(checkpoint),
                    "source_file": filename, "source_sha256": sha256_file(csv_path),
                    "checkpoint_sha256": checkpoint_hash, "readout_layer": entry["readout_layer"],
                    "output_shape": [len(IMAGE_IDS), VOXEL_COUNTS[roi]], "voxel_columns": columns,
                    "voxel_order_sha256": canonical_json_sha256(columns),
                }
                (temp_root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                # A shared backbone must have exactly one readout hook at a time.
                readout.hook.close()
                readout.hook.input = None
                readout.hook._output = None
                del matrix, mapper, logits, readout
                logits = readout = None
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        (temp_root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        verify_completed_triad(temp_root, model, inventory_entry, images_record)
        verify_precision_provenance(provenance, core_root)
        if destination.exists():
            raise GenerationValidationError(f"{model}: a completed directory appeared during generation; refusing overwrite.")
        temp_root.rename(destination)
        return provenance
    except BaseException as exc:
        # Preserve generated portions for inspection, but never promote them as
        # a reference triad or let resume treat them as a completed model.
        partial = locals().get("provenance", {
            "schema_version": 1, "model": model, "dataset": DATASET,
            "images": images_record, "image_ids": list(IMAGE_IDS),
            "image_set_sha256": canonical_json_sha256(images_record), "rois": {},
        })
        partial.update({"status": "incomplete", "error_type": type(exc).__name__, "message": str(exc)})
        (temp_root / "provenance.json").write_text(json.dumps(partial, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        failure_root = output_root / "failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        temp_root.rename(failure_root / f"{model}.{temp_root.name.rsplit('-', 1)[-1]}.partial")
        raise
    finally:
        if readout is not None:
            readout.hook.close()
            readout.hook.input = None
            readout.hook._output = None
        del logits, readout, backbone
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        # Only remove the exact temporary directory created by this invocation.
        if temp_root.exists():
            shutil.rmtree(temp_root)


def core_commit(core_root: Path) -> str | None:
    import subprocess
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=core_root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_failure(output_root: Path, model: str, error: Exception) -> None:
    failures = output_root / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "status": "failed", "error_type": type(error).__name__, "message": str(error),
              "recorded_at": datetime.now(timezone.utc).isoformat(), "substitution": False}
    path = failures / f"{model}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+")
    group.add_argument("--all-missing", action="store_true")
    parser.add_argument("--core-root", type=Path, default=REPO_ROOT.parent / "cortex-core")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/research/cortex-paper/cortex_best_checkpoints"))
    parser.add_argument("--backbone-root", type=Path, default=REPO_ROOT / "artifacts/coverage-expansion/backbone-cache")
    parser.add_argument("--cache-source", type=Path, default=Path("/research/cortex-paper/backbones"))
    parser.add_argument("--prepare-cache", action="store_true", help="Link only readable shared assets into a real-directory local overlay.")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Prepare the local overlay without importing or running models.")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "public/images")
    parser.add_argument("--scores-path", type=Path, default=REPO_ROOT / "public/data/murty185_zero_index.json")
    parser.add_argument("--reference-manifest", type=Path, default=REPO_ROOT / "public/data/reference_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/coverage-expansion/raw-fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly permit missing pretrained-backbone downloads; default is strictly offline.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        parser.error("--device must be cpu or cuda[:index].")
    core_root = args.core_root.resolve()
    inventory = model_inventory(core_root, args.checkpoint_root.resolve(), args.scores_path.resolve(), args.reference_manifest.resolve())
    models = select_models(inventory, args.models, args.all_missing)
    images = image_provenance(args.image_root.resolve())
    if args.prepare_cache or args.prepare_cache_only:
        if args.preflight:
            parser.error("--preflight is read-only; do not combine it with cache preparation.")
        cache_manifest = prepare_backbone_overlay(args.cache_source, args.backbone_root)
        print(json.dumps({"overlay": cache_manifest["overlay_root"], "linked": len(cache_manifest["linked"]),
                          "excluded": len(cache_manifest["excluded"])}, sort_keys=True), flush=True)
        if args.prepare_cache_only:
            return 0
    if args.preflight:
        print(json.dumps({"selected_count": len(models), "image_count": len(images), "image_set_sha256": canonical_json_sha256(images), "models": {m: inventory[m] for m in models}}, sort_keys=True, indent=2))
        return 0 if all(e["readable"] for m in models for e in inventory[m]["rois"].values()) else 1
    output_root = args.output_dir.resolve()
    staging_root = (REPO_ROOT / "artifacts/coverage-expansion").resolve()
    if not output_root.is_relative_to(staging_root) or output_root == staging_root:
        raise GenerationValidationError("Output must be a subdirectory of artifacts/coverage-expansion; live data are never overwritten.")
    offline = not args.allow_network
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_HOME"] = str(args.backbone_root.resolve())
    sys.path.insert(0, str(core_root))
    failed = []
    for i, model in enumerate(models, 1):
        was_completed = (output_root / model).exists()
        print(f"[{i}/{len(models)}] {model}: {'verify existing triad' if was_completed else 'generate NSD-1000 triad'}", flush=True)
        try:
            record = generate_model(model, inventory_entry=inventory[model], core_root=core_root, image_root=args.image_root.resolve(),
                                    backbone_root=args.backbone_root.resolve(), output_root=output_root, images_record=images,
                                    device=args.device, batch_size=args.batch_size, seed=args.seed, offline=offline)
            print(f"  complete: {len(record['rois'])} ROIs; response policy={record['response_policy']}", flush=True)
        except Exception as exc:
            failed.append(model)
            write_failure(output_root, model, exc)
            print(f"  FAILED ({type(exc).__name__}): {exc}", flush=True)
    print(json.dumps({"selected": len(models), "completed": len(models) - len(failed), "failed": failed, "output_dir": str(output_root)}, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
