"""Validate exact model identifiers across InfoViz, Cortex Core, and Cortex App.

The original 26 InfoViz raw-data models remain the default strict subset; an
expanded exact roster must be supplied explicitly. Cortex Core and
Cortex App intentionally expose broader registries; their non-reference models
are reported for review but do not make this check fail.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ROIS = ("ffa", "ppa", "eba")
EXPECTED_MODEL_COUNT = 26
RAW_FILENAME_RE = re.compile(
    r"^murtylab_(?P<model>.+?)_nsd_1000_(?P<roi>ffa|ppa|eba)_(?P<timestamp>\d+Z)\.csv$"
)

# Keep this table empty unless a naming difference is verified and documented.
# Aliases are never guessed and never applied by fuzzy/case-insensitive matching.
VERIFIED_ALIASES: dict[str, str] = {}


class RegistryValidationError(ValueError):
    pass


def parse_raw_models(
    raw_dir: Path, *, expected_models: Sequence[str] | None = None
) -> tuple[set[str], dict[str, list[str]]]:
    if not raw_dir.is_dir():
        raise RegistryValidationError(f"InfoViz raw directory does not exist: {raw_dir}")
    model_rois: dict[str, set[str]] = {}
    unmatched: list[str] = []
    for path in sorted(raw_dir.glob("*.csv"), key=lambda item: item.name):
        match = RAW_FILENAME_RE.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        model_rois.setdefault(match.group("model"), set()).add(match.group("roi"))
    if unmatched:
        raise RegistryValidationError(
            "Unmatched InfoViz raw CSV filenames: " + ", ".join(unmatched)
        )
    if expected_models is not None:
        requested = set(expected_models)
        if not expected_models or len(expected_models) != len(requested):
            raise RegistryValidationError("Explicit model roster must be non-empty and unique")
        if set(model_rois) != requested:
            raise RegistryValidationError(
                "Raw models differ from the explicit roster: "
                f"missing={sorted(requested - set(model_rois))}, "
                f"unexpected={sorted(set(model_rois) - requested)}"
            )
    elif len(model_rois) != EXPECTED_MODEL_COUNT:
        raise RegistryValidationError(
            f"Expected {EXPECTED_MODEL_COUNT} InfoViz models, found {len(model_rois)}"
        )
    incomplete = {
        model: sorted(set(ROIS) - rois)
        for model, rois in sorted(model_rois.items())
        if rois != set(ROIS)
    }
    if incomplete:
        raise RegistryValidationError(f"Incomplete InfoViz model/ROI triads: {incomplete}")
    return set(model_rois), {
        model: sorted(rois, key=ROIS.index) for model, rois in sorted(model_rois.items())
    }


def parse_core_models(core_registry: Path) -> set[str]:
    try:
        tree = ast.parse(core_registry.read_text(encoding="utf-8"), filename=str(core_registry))
    except (OSError, SyntaxError) as exc:
        raise RegistryValidationError(f"Could not parse Cortex Core registry: {exc}") from exc

    dictionary: ast.Dict | None = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "backbone_models" for target in targets):
            value = node.value
            if isinstance(value, ast.Dict):
                dictionary = value
            break
    if dictionary is None:
        raise RegistryValidationError(
            f"Could not find a literal backbone_models dictionary in {core_registry}"
        )

    names: list[str] = []
    for value in dictionary.values:
        if not isinstance(value, ast.Dict):
            raise RegistryValidationError("Every Cortex Core backbone entry must be a dictionary")
        entry: dict[str, Any] = {}
        for key_node, value_node in zip(value.keys, value.values):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if isinstance(value_node, ast.Constant):
                    entry[key_node.value] = value_node.value
        name = entry.get("name")
        if not isinstance(name, str):
            raise RegistryValidationError("A Cortex Core backbone entry has no literal string name")
        names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RegistryValidationError(f"Duplicate Cortex Core model names: {duplicates}")
    return set(names)


def validate_model_identifiers(
    models: Sequence[str], *, core_registry: Path, app_constants: Path
) -> tuple[str, ...]:
    """Validate an explicit coverage roster without guessing aliases or counts."""
    requested = tuple(models)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(not isinstance(model, str) or not re.fullmatch(r"[a-z0-9_]+(?:\.[0-9]+)?", model) for model in requested)
    ):
        raise RegistryValidationError("Explicit model roster must contain unique safe exact identifiers")
    core_models = parse_core_models(core_registry)
    app_models = parse_app_models(app_constants)
    missing_core = sorted(set(requested) - core_models)
    missing_app = sorted(set(requested) - app_models)
    if missing_core or missing_app:
        raise RegistryValidationError(
            "Explicit reference model identifiers are missing from sibling registries: "
            f"Core={missing_core or 'none'}, App={missing_app or 'none'}"
        )
    return tuple(sorted(requested))


def _javascript_array(text: str, variable_name: str, path: Path) -> str:
    marker = re.search(
        rf"(?:export\s+)?const\s+{re.escape(variable_name)}\s*=\s*\[",
        text,
    )
    if marker is None:
        raise RegistryValidationError(f"Could not find {variable_name} array in {path}")
    start = marker.end() - 1
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise RegistryValidationError(f"Unterminated {variable_name} array in {path}")


def parse_app_models(app_constants: Path) -> set[str]:
    try:
        array_text = _javascript_array(
            app_constants.read_text(encoding="utf-8"),
            "MODEL_OPTIONS",
            app_constants,
        )
    except OSError as exc:
        raise RegistryValidationError(f"Could not read Cortex App MODEL_OPTIONS: {exc}") from exc
    names = re.findall(
        r"[\"']value[\"']\s*:\s*[\"']([a-zA-Z0-9_.-]+)[\"']",
        array_text,
    )
    if not names:
        raise RegistryValidationError(f"No literal model values found in {app_constants}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RegistryValidationError(f"Duplicate Cortex App MODEL_OPTIONS values: {duplicates}")
    return set(names)


def validate_registries(
    *,
    raw_dir: Path,
    core_registry: Path,
    app_constants: Path,
    expected_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    raw_models, triads = parse_raw_models(raw_dir, expected_models=expected_models)
    core_models = parse_core_models(core_registry)
    app_models = parse_app_models(app_constants)

    missing_from_core = sorted(raw_models - core_models)
    missing_from_app = sorted(raw_models - app_models)
    report = {
        "valid": not missing_from_core and not missing_from_app,
        "policy": "InfoViz reference models must be an exact-name subset of Core and App",
        "verified_aliases": VERIFIED_ALIASES,
        "counts": {
            "infoviz_reference": len(raw_models),
            "cortex_core": len(core_models),
            "cortex_app": len(app_models),
        },
        "supported_models": sorted(raw_models),
        "infoviz_triads": triads,
        "strict_subset_errors": {
            "missing_from_core": missing_from_core,
            "missing_from_app": missing_from_app,
        },
        "broader_registry_differences": {
            "core_without_reference_data": sorted(core_models - raw_models),
            "app_without_reference_data": sorted(app_models - raw_models),
            "core_not_in_app": sorted(core_models - app_models),
            "app_not_in_core": sorted(app_models - core_models),
        },
    }
    if not report["valid"]:
        raise RegistryValidationError(
            "Reference model identifiers are missing from sibling registries: "
            f"Core={missing_from_core or 'none'}, App={missing_from_app or 'none'}"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+",
        help="Explicit complete reference model roster; omit for the original strict 26-model contract",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "image_by_model",
    )
    parser.add_argument(
        "--core-registry",
        type=Path,
        default=REPO_ROOT.parent / "cortex-core" / "cortex" / "models" / "__init__.py",
    )
    parser.add_argument(
        "--app-constants",
        type=Path,
        default=REPO_ROOT.parent / "cortex-app" / "assets" / "js" / "constants.jsx",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete JSON report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = validate_registries(
            raw_dir=args.raw_dir,
            core_registry=args.core_registry,
            app_constants=args.app_constants,
            expected_models=args.models,
        )
    except RegistryValidationError as exc:
        raise SystemExit(f"Model registry validation failed: {exc}") from exc
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    counts = report["counts"]
    differences = report["broader_registry_differences"]
    print(
        "Exact supported subset valid: "
        f"InfoViz={counts['infoviz_reference']}, Core={counts['cortex_core']}, "
        f"App={counts['cortex_app']}"
    )
    print(
        "Broader differences (reported, not failures): "
        f"Core-only-vs-App={len(differences['core_not_in_app'])}, "
        f"App-only-vs-Core={len(differences['app_not_in_core'])}"
    )


if __name__ == "__main__":
    main()
