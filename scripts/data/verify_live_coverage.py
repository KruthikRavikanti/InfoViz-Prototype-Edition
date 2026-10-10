#!/usr/bin/env python3
"""Check real HTTP upload sessions against the published reference predictions.

Uploads are copies of three public canonical images, never a user's images.
Stochastic models retain structural/ranking checks without requiring equality
to a different stochastic draw. Tokens stay in a local ignored QA report.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parents[2]
# Resolve from this module, not REPO_ROOT: fixtures may replace the data root.
CORE_ROOT = Path(__file__).resolve().parents[3] / "cortex-core"
ROIS = ("ffa", "ppa", "eba")
COUNTS = {"ffa": 156, "ppa": 170, "eba": 518}
IMAGE_IDS = ("image_001.png", "image_042.png", "image_185.png")
REFERENCE_IDS = tuple(f"image_{i:03d}.png" for i in range(1, 186))
GOLDEN_TOLERANCE = 5e-4
JOINT_METADATA = {
    "schema_version": 1, "method": "hierarchical_correlation",
    "distance_metric": "correlation", "linkage_method": "average",
    "normalization": "reference_zscore",
}


class LiveCoverageVerificationError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        # Session IDs and detailed HTTP errors belong only in the ignored report.
        super().__init__("Live verification failed; inspect the local QA report.")


def normalized_euclidean_rdm(matrix: np.ndarray) -> np.ndarray:
    """Match Core utils/rdm.py: raw Euclidean distance / full voxel count."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Reference response matrix must be finite and two-dimensional")
    return squareform(pdist(values, metric="euclidean")) / values.shape[1]


def rdm_absolute_tolerance(voxel_count: int) -> float:
    # If each predicted voxel differs by at most epsilon, each pairwise
    # difference differs by at most 2*epsilon. The reverse triangle inequality
    # bounds normalized Euclidean distance drift by 2*epsilon/sqrt(voxel_count).
    if voxel_count <= 0:
        raise ValueError("RDM voxel count must be positive")
    return float(2 * GOLDEN_TOLERANCE / np.sqrt(voxel_count))


def select_qa_models(reference: dict, models: list[str] | None, all_new: bool,
                     baseline_models: list[str] | None = None, *,
                     all_supported: bool = False) -> list[str]:
    if sum((bool(models), all_new, all_supported)) != 1 or (baseline_models and not all_new):
        raise ValueError("Choose --models, --all-new, or --all-supported; --baseline-models requires --all-new")
    selected = (reference["supported_models"] if all_supported else
                models if models else reference.get("coverage_extension", {}).get("added_models", []))
    if not selected:
        raise ValueError("Published reference has no added models to verify")
    selected = list(dict.fromkeys([*selected, *(baseline_models or [])]))
    if not set(selected).issubset(reference["supported_models"]):
        raise ValueError("Requested QA models are not all in the published reference roster")
    return selected


def prediction_policies(reference: dict) -> dict:
    """Read the release's actual noise policy, never infer it from model names."""
    standardization = reference.get("prediction_standardization")
    if standardization is not None:
        if "coverage_extension" in reference:
            raise ValueError("Published reference has ambiguous release lineage")
        policies = standardization.get("model_prediction_policies", {})
        if (standardization.get("policy") != "full_roster_fp32_regeneration_v1"
                or standardization.get("regenerated_models") != reference["supported_models"]
                or set(policies) != set(reference["supported_models"])):
            raise ValueError("Published standardization has incomplete prediction policies")
        return policies
    return reference.get("coverage_extension", {}).get("model_prediction_policies", {})


@lru_cache(maxsize=1)
def joint_hierarchy_schema():
    """Import Core's structural tree validator without loading inference models."""
    sys.path.insert(0, str(CORE_ROOT))
    from cortex.comparison.schemas import JointHierarchy
    return JointHierarchy


def validate_joint_hierarchy(joint: dict, stimulus_ids: list[str], *, context: str) -> None:
    # Require transmitted metadata explicitly: Pydantic defaults must not turn
    # omitted method/normalization fields into apparently valid HTTP evidence.
    if (not isinstance(joint, dict) or type(joint.get("schema_version")) is not int
            or any(joint.get(key) != expected for key, expected in JOINT_METADATA.items())):
        raise ValueError(f"{context}: joint hierarchy has missing or incompatible method metadata")
    try:
        joint_hierarchy_schema().model_validate(joint)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid joint hierarchy tree/cluster contract") from exc
    if (joint["reference_count"] != len(REFERENCE_IDS)
            or joint["uploaded_count"] != len(stimulus_ids)
            or [m["id"] for m in joint["members"] if m["kind"] == "reference"] != list(REFERENCE_IDS)
            or [m["id"] for m in joint["members"] if m["kind"] == "uploaded"] != stimulus_ids):
        raise ValueError(f"{context}: joint hierarchy changed reference/upload identity or order")


def load_reference_means(entry: dict, *, context: str) -> np.ndarray:
    """Read the saved ranking means, not recomputed raw-CSV voxel averages."""
    root = REPO_ROOT / "artifacts/reference"
    relative = Path(entry["path"])
    path = root / relative
    if (relative.is_absolute() or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()):
        raise ValueError(f"{context}: unsafe private reference artifact path")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"{context}: private reference artifact checksum changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["image_ids"].tolist() != list(REFERENCE_IDS):
            raise ValueError(f"{context}: private reference image identity/order changed")
        means = saved["reference_means"].copy()
    if (means.dtype != np.dtype("float64") or means.shape != (len(REFERENCE_IDS),)
            or not np.isfinite(means).all()):
        raise ValueError(f"{context}: malformed saved reference ranking means")
    return means


def validate_ranking_statistics(reference_means: np.ndarray, stimuli: list[dict], *, context: str) -> None:
    """Independently check live statistics, including draws from noisy models.

    Only the returned live mean is ranked against the saved reference means.
    No regenerated query rank or equality to a different noisy draw is assumed.
    """
    values = np.asarray(reference_means, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not stimuli:
        raise ValueError(f"{context}: ranking inputs must be finite and nonempty")
    means = []
    for stimulus in stimuli:
        mean = stimulus.get("mean")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not np.isfinite(mean):
            raise ValueError(f"{context}: live ranking mean is missing or non-finite")
        means.append(float(mean))
    # Descending combined ordering, references first on equal means, then
    # original image/upload order. Do not import the production ranking helper.
    items = [(float(value), 0, index) for index, value in enumerate(values)]
    items.extend((value, 1, index) for index, value in enumerate(means))
    items.sort(key=lambda item: (-item[0], item[1], item[2]))
    session_ranks = {item[2]: rank for rank, item in enumerate(items, 1) if item[1] == 1}
    session_count = len(items)
    extreme_count = min(10, len(values))
    for index, (stimulus, mean) in enumerate(zip(stimuli, means)):
        greater = int(np.count_nonzero(values > mean))
        equal = int(np.count_nonzero(values == mean))
        less = int(np.count_nonzero(values < mean))
        reference_rank = float(greater + 1) if equal == 0 else float(greater + (equal + 1) / 2)
        percentile = 100.0 * (less + equal / 2.0) / len(values)
        for key, expected in (("reference_rank", reference_rank), ("reference_percentile", percentile)):
            actual = stimulus.get(key)
            if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual)
                    or not np.isclose(actual, expected, rtol=0, atol=1e-12)):
                raise ValueError(f"{context}: incorrect live {key} for upload {index}")
        rank = session_ranks[index]
        for key, expected in (("session_rank", rank), ("session_count", session_count)):
            if type(stimulus.get(key)) is not int or stimulus[key] != expected:
                raise ValueError(f"{context}: incorrect live {key} for upload {index}")
        for key, expected in (("enters_top", rank <= extreme_count),
                              ("enters_bottom", rank > session_count - extreme_count)):
            if type(stimulus.get(key)) is not bool or stimulus[key] is not expected:
                raise ValueError(f"{context}: incorrect live {key} for upload {index}")
        closest = sorted(range(len(values)), key=lambda i: (abs(float(values[i]) - mean), i))[:5]
        neighbors = stimulus.get("closest_by_mean")
        if (not isinstance(neighbors, list)
                or [neighbor.get("image_name") for neighbor in neighbors] != [REFERENCE_IDS[i] for i in closest]):
            raise ValueError(f"{context}: incorrect closest-by-mean reference ordering for upload {index}")


def verify_live(models: list[str], core_url: str, upload_root: Path) -> dict:
    report = {"reference_version": None, "requested_models": models, "checked_models": [],
              "sessions": [], "golden_tolerance": GOLDEN_TOLERANCE, "status": "running"}
    http = None
    temporary = None
    current_model = None
    try:
        reference = json.loads((REPO_ROOT / "artifacts/reference/manifest.json").read_text())
        version = report["reference_version"] = reference["reference_version"]
        policies = prediction_policies(reference)
        if reference.get("dataset") != "nsd_1000" or reference.get("reference_count") != len(REFERENCE_IDS):
            raise ValueError("Published reference is not the canonical NSD-1000/Murty185 bundle")
        if not models or not set(models).issubset(reference["supported_models"]):
            raise ValueError("Requested QA models are not all in the published reference roster")
        http = requests.Session()
        capabilities = http.get(f"{core_url}/comparison/capabilities", timeout=30)
        capabilities.raise_for_status()
        advertised = capabilities.json()
        if (advertised.get("available") is not True
                or advertised.get("reference_version") != version
                or advertised.get("supported_models") != reference["supported_models"]
                or advertised.get("reference_count") != len(REFERENCE_IDS)
                or advertised.get("dataset") != "nsd_1000"
                or advertised.get("comparison_dataset") != "nsd_1000"
                or advertised.get("rois") != list(ROIS)):
            raise ValueError("Running Core is not serving the exact available canonical reference identity")
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="coverage-qa-", dir=upload_root))
        for image_id in IMAGE_IDS:
            shutil.copyfile(REPO_ROOT / "public/images" / image_id, temporary / image_id)
        for model in models:
            current_model = model
            attempt = {"model": model, "session_id": None, "status": "running"}
            report["sessions"].append(attempt)
            print(f"Checking real HTTP session: {model}", flush=True)
            response = http.post(f"{core_url}/comparison-sessions", json={
                "model": model,
                "stimuli": [{"path": str(temporary / image_id), "display_name": image_id} for image_id in IMAGE_IDS],
                "source_request": {"model": model, "dataset": "nsd_1000", "roi": "ffa"},
            }, timeout=300)
            response.raise_for_status()
            token = attempt["session_id"] = response.json()["session_id"]
            if not isinstance(token, str) or not token:
                raise ValueError(f"{model}: session creation returned no valid session ID")
            fetched = http.get(f"{core_url}/comparison-sessions/{token}", timeout=30)
            fetched.raise_for_status()
            session = fetched.json()
            if (session.get("schema_version") != 1
                    or session["source_request"] != {"model": model, "dataset": "nsd_1000", "roi": "ffa"}
                    or session["comparison"]["model"] != model
                    or session["comparison"]["reference_version"] != version
                    or session["comparison"]["dataset"] != "nsd_1000"
                    or session["comparison"]["rois"] != list(ROIS)
                    or session["comparison"]["reference_count"] != len(REFERENCE_IDS)
                    or session.get("brain_map_results") is None):
                raise ValueError(f"{model}: incorrect session or absent audited Subject 1 brain payload")
            stimuli = session["stimuli"]
            stimulus_ids = [stimulus["id"] for stimulus in stimuli]
            brain = session["brain_map_results"]
            if ([stimulus["display_name"] for stimulus in stimuli] != list(IMAGE_IDS)
                    or len(set(stimulus_ids)) != len(IMAGE_IDS)
                    or any(not isinstance(value, str) or not value for value in stimulus_ids)
                    or brain.get("subject_id") != "s1"
                    or brain.get("response_space") != "nsd_1000_subject1_floc_t7_v1"
                    or brain.get("roi_voxel_counts") != COUNTS
                    or [stimulus["stimulus_id"] for stimulus in brain["stimuli"]] != stimulus_ids):
                raise ValueError(f"{model}: stimulus image identity/order or Subject 1 response identity changed")
            stochastic = policies.get(model, {}).get("response_policy") == "seeded_stochastic_sampling"
            max_drift = 0.0
            max_rdm_drift = 0.0
            rdm_tolerances = {}
            for roi in ROIS:
                result = session["roi_results"][roi]
                joint = result["joint_hierarchy"]
                validate_joint_hierarchy(joint, stimulus_ids, context=f"{model}/{roi}")
                if [s["stimulus_id"] for s in result["stimuli"]] != stimulus_ids:
                    raise ValueError(f"{model}/{roi}: ROI results changed upload identity or order")
                entry = reference["artifacts"][model][roi]
                reference_means = load_reference_means(entry, context=f"{model}/{roi}")
                validate_ranking_statistics(reference_means, result["stimuli"], context=f"{model}/{roi}")
                frame = pd.read_csv(REPO_ROOT / "data/image_by_model" / entry["source_file"], index_col=0)
                if result["voxel_count"] != frame.shape[1]:
                    raise ValueError(f"{model}/{roi}: wrong full pooled readout width")
                if list(frame.index) != list(REFERENCE_IDS):
                    raise ValueError(f"{model}/{roi}: reference CSV image identity/order changed")
                expected_matrix = frame.loc[list(IMAGE_IDS)].to_numpy(dtype=np.float32)
                if not np.isfinite(expected_matrix).all():
                    raise ValueError(f"{model}/{roi}: reference predictions are not finite")
                actual_rdm = np.asarray(result["rdm"], dtype=np.float64)
                if (actual_rdm.shape != (len(IMAGE_IDS), len(IMAGE_IDS))
                        or not np.isfinite(actual_rdm).all() or (actual_rdm < 0).any()
                        or not np.allclose(actual_rdm, actual_rdm.T, atol=1e-12, rtol=0)
                        or not np.allclose(np.diag(actual_rdm), 0, atol=1e-12, rtol=0)):
                    raise ValueError(f"{model}/{roi}: malformed full pooled response RDM")
                rdm_tolerance = rdm_tolerances[roi] = float(rdm_absolute_tolerance(frame.shape[1]))
                if not stochastic:
                    expected_rdm = normalized_euclidean_rdm(expected_matrix)
                    rdm_drift = float(np.max(np.abs(actual_rdm - expected_rdm)))
                    max_rdm_drift = max(max_rdm_drift, rdm_drift)
                    if rdm_drift > rdm_tolerance:
                        raise ValueError(f"{model}/{roi}: full pooled response RDM drift {rdm_drift} exceeds tolerance {rdm_tolerance}")
                for index, image_id in enumerate(IMAGE_IDS):
                    expected = expected_matrix[index]
                    actual = np.asarray(session["brain_map_results"]["stimuli"][index]["roi_responses"][roi], dtype=np.float32)
                    if (actual.shape != (COUNTS[roi],) or not np.isfinite(actual).all()
                            or not np.isfinite(result["stimuli"][index]["mean"])
                            or not np.isfinite(result["stimuli"][index]["sem"])):
                        raise ValueError(f"{model}/{roi}: malformed or non-finite Subject 1 responses")
                    if not stochastic:
                        drift = float(np.max(np.abs(actual - expected[:COUNTS[roi]])))
                        max_drift = max(max_drift, drift)
                        if drift > GOLDEN_TOLERANCE or abs(result["stimuli"][index]["mean"] - expected.mean(dtype=np.float64)) > GOLDEN_TOLERANCE:
                            raise ValueError(f"{model}/{roi}/{image_id}: live/staged prediction drift exceeds the existing golden tolerance ({drift})")
                        if result["stimuli"][index]["closest_in_voxel_space"][0]["image_name"] != image_id:
                            raise ValueError(f"{model}/{roi}: canonical image did not match itself in voxel space")
            for stimulus in session["stimuli"]:
                preview = http.get(core_url + stimulus["image_url"], timeout=30)
                preview.raise_for_status()
                if not preview.headers.get("content-type", "").startswith("image/"):
                    raise ValueError(f"{model}: uploaded preview is not an actual image response")
            report["checked_models"].append({"model": model, "session_id": token, "image_count": len(IMAGE_IDS),
                            "rois": list(ROIS), "brain_map_available": True,
                            "stochastic": stochastic, "maximum_subject1_voxel_drift": None if stochastic else max_drift,
                            "maximum_full_pooled_rdm_drift": None if stochastic else max_rdm_drift,
                            "rdm_absolute_tolerances": rdm_tolerances,
                            "ranking_statistics_checked": True, "joint_tree_contract_checked": True})
            attempt["status"] = "passed"
            print(f"  passed: 3 ROIs, full joint tree, rankings, previews, Subject 1; stochastic={stochastic}; drift={max_drift:.3g}", flush=True)
        report["status"] = "passed"
        return report
    except Exception as exc:
        report.update({"status": "failed", "failed_model": current_model,
                       "error_type": type(exc).__name__, "error": str(exc)})
        if report["sessions"] and report["sessions"][-1]["status"] == "running":
            report["sessions"][-1]["status"] = "failed"
        raise LiveCoverageVerificationError(report) from exc
    finally:
        if http is not None:
            http.close()
        if temporary is not None:
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+")
    group.add_argument("--all-new", action="store_true", help="Sequentially verify every added model in the published release")
    group.add_argument("--all-supported", action="store_true", help="Sequentially verify the complete published runnable model roster")
    parser.add_argument("--baseline-models", nargs="+", help="Also check these original models when using --all-new")
    parser.add_argument("--core-url", default="http://127.0.0.1:5558")
    parser.add_argument("--upload-root", type=Path, default=REPO_ROOT.parent / ".runtime/gradio")
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    allowed = (REPO_ROOT / "artifacts/coverage-expansion", REPO_ROOT / "artifacts/precision-standardization")
    target = args.report_path.resolve()
    if (not any(target.is_relative_to(root.resolve()) and target != root.resolve() for root in allowed)
            or args.report_path.exists()):
        parser.error("QA report must be a new file inside an approved ignored release-audit directory")
    try:
        reference = json.loads((REPO_ROOT / "artifacts/reference/manifest.json").read_text())
        models = select_qa_models(reference, args.models, args.all_new, args.baseline_models,
                                 all_supported=args.all_supported)
        report = verify_live(models, args.core_url.rstrip("/"), args.upload_root.resolve())
    except LiveCoverageVerificationError as exc:
        report = exc.report
    except Exception as exc:
        report = {"status": "failed", "checked_models": [], "sessions": [],
                  "error_type": type(exc).__name__, "error": str(exc)}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if report["status"] != "passed":
        print(f"Live verification failed after {len(report['checked_models'])} models; report saved to {args.report_path}")
        raise SystemExit(1)
    print(f"Live verification passed for {len(report['checked_models'])} models; report saved to {args.report_path}")


if __name__ == "__main__":
    main()
