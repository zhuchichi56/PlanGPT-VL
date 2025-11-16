#!/usr/bin/env python
"""
Image Filtering Script

Filters images to identify valid planning maps.

Usage:
    python -m scripts.filter_images --input_dir /path/to/images --output refined_results.json
"""

import os
import sys
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.image_utils import process_image_directory
from common.io_utils import save_json
from filtering.planning_map_filter import filter_planning_maps


def main(input_dir: str,
         output_path: str,
         batch_size: int = 500,
         checkpoint_dir: str = None):
    """
    Filter images for valid planning maps

    Args:
        input_dir: Directory containing images
        output_path: Path to save filter results
        batch_size: Batch size (default: 500)
        checkpoint_dir: Checkpoint directory
    """
    # Get all images
    image_paths = process_image_directory(input_dir)
    print(f"Found {len(image_paths)} images")

    # Filter
    results = filter_planning_maps(
        image_paths,
        batch_size=batch_size,
        output_dir=checkpoint_dir
    )

    # Save
    save_json(results, output_path)

    # Statistics
    planning_count = sum(1 for r in results if r['is_planning_map'] == 1)
    print(f"\nResults:")
    print(f"  Total images: {len(results)}")
    print(f"  Planning maps: {planning_count}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
