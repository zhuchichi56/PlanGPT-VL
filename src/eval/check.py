#!/usr/bin/env python3
"""Validate that every image referenced in planbench-subset.json exists locally.

The script is intentionally lightweight so it can be run before evaluation.
It checks every `image_url` entry and reports the paths that are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "planbench-subset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the dataset json file (default: %(default)s).",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Base directory that contains the referenced images. "
            "Defaults to the dataset's parent directory."
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Iterable[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Dataset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse json file {path}: {exc}") from exc


def resolve_image_path(image_root: Path, image_url: str) -> Path:
    image_path = Path(image_url)
    if image_path.is_absolute():
        return image_path
    return image_root / image_path


def check_images(
    entries: Iterable[dict], image_root: Path
) -> List[Tuple[int, str, Path]]:
    missing = []
    for idx, entry in enumerate(entries, start=1):
        image_url = entry.get("image_url")
        if not image_url:
            missing.append((idx, "<missing image_url>", Path("<none>")))
            continue
        image_path = resolve_image_path(image_root, image_url)
        if not image_path.is_file():
            missing.append((idx, image_url, image_path))
    return missing


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    image_root = Path(args.image_root) if args.image_root else dataset_path.parent

    entries = load_dataset(dataset_path)
    missing = check_images(entries, image_root)

    if not missing:
        print("All referenced images exist.")
        return

    print(f"Missing {len(missing)} image(s):")
    for idx, image_url, missing_path in missing:
        print(f"  #{idx}: {image_url} -> {missing_path}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
