"""
Resolution Filter

Filters images based on resolution criteria.
"""

import os
import re
import shutil
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from loguru import logger
import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.io_utils import load_json, save_json


class ResolutionFilter:
    """Filter images by resolution"""

    def __init__(self, filter_percent: float = 5.0):
        """
        Initialize resolution filter

        Args:
            filter_percent: Percentage of highest resolution images to filter out
        """
        self.filter_percent = filter_percent

    def _get_resolution(self, image_path: str) -> Optional[int]:
        """Get image resolution (height * width)"""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img.shape[0] * img.shape[1]
        except Exception as e:
            logger.error(f"Error reading {image_path}: {e}")
        return None

    def filter(self,
              input_data: List[Dict],
              output_images_dir: str,
              output_text_dir: str = None,
              original_text_dir: str = None,
              max_workers: int = 64) -> List[Dict]:
        """
        Filter images by resolution

        Args:
            input_data: List of dictionaries with 'image' and 'is_planning_map' fields
            output_images_dir: Directory for filtered images
            output_text_dir: Directory for associated text files
            original_text_dir: Directory containing original text files
            max_workers: Maximum worker threads

        Returns:
            List of filtered image data
        """
        # Create output directories
        os.makedirs(output_images_dir, exist_ok=True)
        if output_text_dir:
            os.makedirs(output_text_dir, exist_ok=True)

        # Build text file map
        text_file_map = {}
        if original_text_dir and os.path.exists(original_text_dir):
            try:
                for filename in os.listdir(original_text_dir):
                    if filename.endswith('.txt'):
                        hash_code = os.path.splitext(filename)[0]
                        text_file_map[hash_code] = os.path.join(original_text_dir, filename)
            except Exception as e:
                logger.error(f"Error accessing text directory: {e}")

        # Filter planning maps
        planning_maps = [item for item in input_data if item.get("is_planning_map") == 1]
        logger.info(f"Found {len(planning_maps)} planning maps")

        # Get resolutions
        image_resolutions = []
        lock = threading.Lock()

        def process_image(item):
            resolution = self._get_resolution(item["image"])
            if resolution:
                with lock:
                    image_resolutions.append((item, resolution))
            else:
                logger.warning(f"Could not read image: {item['image']}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_image, planning_maps),
                     total=len(planning_maps),
                     desc="Processing images"))

        # Sort by resolution
        image_resolutions.sort(key=lambda x: x[1], reverse=True)

        # Calculate filter count
        filter_count = int(len(image_resolutions) * self.filter_percent / 100)
        logger.info(f"Filtering out {filter_count} images with highest resolution ({self.filter_percent}%)")

        # Keep remaining images
        filtered_maps = [item[0] for item in image_resolutions[filter_count:]]
        logger.info(f"Remaining {len(filtered_maps)} planning maps after filtering")

        # Log resolution stats
        if image_resolutions:
            original_resolutions = [res for _, res in image_resolutions]
            filtered_resolutions = [res for _, res in image_resolutions[filter_count:]]
            logger.info(f"Original - Avg resolution: {sum(original_resolutions)/len(original_resolutions):.2f}, "
                       f"Max: {max(original_resolutions)}")
            logger.info(f"Filtered - Avg resolution: {sum(filtered_resolutions)/len(filtered_resolutions):.2f}, "
                       f"Max: {max(filtered_resolutions)}")

        # Copy files
        processed_count = 0
        hash_code_set = set()

        for item in filtered_maps:
            try:
                image_path = item["image"]
                image_filename = os.path.basename(image_path)

                # Extract hash code
                match = re.match(r'([a-f0-9]+)_figure\d+\.\w+', image_filename)
                if match:
                    hash_code = match.group(1)
                    hash_code_set.add(hash_code)
                else:
                    logger.warning(f"Cannot extract hash code from: {image_filename}")
                    continue

                # Copy image
                dest_image_path = os.path.join(output_images_dir, image_filename)
                shutil.copy2(image_path, dest_image_path)

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} planning maps")

            except Exception as e:
                logger.error(f"Error processing image {image_path if 'image_path' in locals() else 'unknown'}: {e}")

        # Copy text files
        if output_text_dir and text_file_map:
            text_copied_count = 0
            for hash_code in hash_code_set:
                if hash_code in text_file_map:
                    try:
                        src_text_path = text_file_map[hash_code]
                        dest_text_path = os.path.join(output_text_dir, f"{hash_code}.txt")
                        shutil.copy2(src_text_path, dest_text_path)
                        text_copied_count += 1
                    except Exception as e:
                        logger.error(f"Error copying text file {hash_code}.txt: {e}")
                else:
                    logger.warning(f"No original text file found for hash code {hash_code}")

            logger.info(f"Copied {text_copied_count} text files")

        logger.info(f"Processing completed. Total processed {processed_count} planning maps")

        return filtered_maps


def filter_by_resolution(input_json: str,
                        original_data_dir: str,
                        output_dir: str,
                        filter_percent: float = 5.0):
    """
    Convenience function to filter by resolution

    Args:
        input_json: Path to input JSON with filter results
        original_data_dir: Directory with original images and text
        output_dir: Output directory
        filter_percent: Percentage to filter
    """
    data = load_json(input_json)

    output_images_dir = os.path.join(output_dir, "images")
    output_text_dir = os.path.join(output_dir, "text")
    original_text_dir = os.path.join(original_data_dir, "text")

    filter_obj = ResolutionFilter(filter_percent)
    filtered_data = filter_obj.filter(
        data,
        output_images_dir,
        output_text_dir,
        original_text_dir
    )

    # Save summary
    summary_path = os.path.join(output_dir, "planning_maps_summary.json")
    save_json(filtered_data, summary_path)
    logger.info(f"Summary saved to: {summary_path}")

    return filtered_data
