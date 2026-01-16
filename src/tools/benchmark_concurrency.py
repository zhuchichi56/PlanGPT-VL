#!/usr/bin/env python
"""
Benchmark inference throughput across different concurrency levels.
"""

import argparse
import os
import time
from typing import List

from pipeline import generate_questions
from tools.inference_utils import InferenceParams


def load_images(image_dir: str, limit: int) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    files = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(exts)
    ]
    files.sort()
    if limit:
        files = files[:limit]
    return files


def run_once(image_paths: List[str], workers: int) -> float:
    params = InferenceParams(
        model=os.environ.get("VLLM_MODEL"),
        server_url=os.environ.get("VLLM_API_BASE", "http://localhost:8000"),
        max_workers=workers,
    )
    start = time.time()
    _ = generate_questions(
        image_paths,
        inference_params=params,
    )
    return time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference concurrency")
    parser.add_argument("--image_dir", required=True, help="Directory with images")
    parser.add_argument("--limit", type=int, default=6, help="Number of images to use")
    parser.add_argument("--workers", default="1,2,4,8", help="Comma-separated worker counts")
    args = parser.parse_args()

    images = load_images(args.image_dir, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    print(f"Images: {len(images)}")
    for w in [int(x) for x in args.workers.split(",") if x.strip()]:
        elapsed = run_once(images, w)
        print(f"workers={w} elapsed={elapsed:.2f}s avg_per_image={elapsed/len(images):.2f}s")


if __name__ == "__main__":
    main()
