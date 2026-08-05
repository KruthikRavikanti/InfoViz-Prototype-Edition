#!/usr/bin/env python3
"""
DreamSim embedding reader for this repo.

The precomputed bundle lives at data/dreamsim/dreamsim_embeddings.pth.
The clustering pipeline uses the "murty185" key, whose 185 rows line up with
public/images/image_001.png through public/images/image_185.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PTH = REPO_ROOT / "data" / "dreamsim" / "dreamsim_embeddings.pth"


def load_dreamsim_embeddings(
    path: Path | str = DEFAULT_PTH,
    keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load DreamSim stimulus embeddings.

    Returns a dict where each value is a tensor shaped (N, 1792).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(path), map_location="cpu")

    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict[str, Tensor], got {type(blob)}")

    if keys is None:
        return blob

    missing = [key for key in keys if key not in blob]
    if missing:
        raise KeyError(f"Missing keys {missing}. Available keys: {sorted(blob.keys())}")

    return {key: blob[key] for key in keys}


def print_embedding_summary(embeddings: Dict[str, torch.Tensor]) -> None:
    print(f"{'key':<16} {'shape':<16} {'dtype'}")
    print("-" * 48)
    for key in sorted(embeddings.keys()):
        value = embeddings[key]
        if isinstance(value, torch.Tensor):
            print(f"{key:<16} {str(tuple(value.shape)):<16} {value.dtype}")
        else:
            print(f"{key:<16} {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the repo's DreamSim embedding bundle.")
    parser.add_argument("--pth", type=Path, default=DEFAULT_PTH)
    parser.add_argument("--keys", nargs="+", default=None)
    parser.add_argument(
        "--save-subset",
        type=Path,
        default=None,
        help="Optional output .pth containing only --keys.",
    )
    args = parser.parse_args()

    embeddings = load_dreamsim_embeddings(args.pth, keys=args.keys)
    print_embedding_summary(embeddings)

    if args.save_subset is not None:
        if args.keys is None:
            raise SystemExit("--save-subset requires --keys")
        args.save_subset.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, str(args.save_subset))
        print(f"Wrote {args.save_subset}")


if __name__ == "__main__":
    main()
