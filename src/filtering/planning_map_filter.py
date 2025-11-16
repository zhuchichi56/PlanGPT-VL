"""
Planning Map Filter

Filters images to identify valid urban planning maps.
"""

import os
import re
from typing import List, Dict, Tuple
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.prompts import PROMPTS
from common.inference_utils import parallel_image_inference_batch
from common.io_utils import save_json


class PlanningMapFilter:
    """Filter for identifying valid planning maps"""

    def __init__(self, prompt_key: str = "planning_map_filter"):
        """
        Initialize filter

        Args:
            prompt_key: Key for filtering prompt
        """
        self.prompt_key = prompt_key
        self.prompt = PROMPTS[prompt_key]

    def _parse_result(self, text: str) -> Tuple[str, int]:
        """
        Extract analysis and score from VLM response

        Args:
            text: Response text

        Returns:
            Tuple of (analysis, score)
        """
        # Extract analysis
        analysis = ""
        if match := re.search(r'分析：(.*?)(?=判断：|\\boxed{|$)', text, re.DOTALL):
            analysis = match.group(1).strip()

        # Extract score
        score = 0
        if match := re.search(r'\\boxed{(\d+)}', text):
            try:
                score = int(match.group(1))
            except ValueError:
                logger.warning(f"Failed to parse score from: {match.group(1)}")

        # Fallback: use text before boxed as analysis
        if not analysis and text:
            parts = re.split(r'\\boxed{\d+}', text)
            if parts and parts[0].strip():
                analysis = parts[0].strip()

        return analysis, score

    def filter(self,
              image_paths: List[str],
              batch_size: int = 500,
              output_dir: str = None) -> List[Dict]:
        """
        Filter images for valid planning maps

        Args:
            image_paths: List of image paths to filter
            batch_size: Batch size for processing
            output_dir: Output directory for results

        Returns:
            List of dictionaries with keys: 'image', 'analysis', 'is_planning_map'
        """
        logger.info(f"Filtering {len(image_paths)} images")

        # Prepare prompts
        prompts = [self.prompt] * len(image_paths)

        # Run inference
        results = parallel_image_inference_batch(
            prompts,
            image_paths,
            batch_size=batch_size,
            output_dir=output_dir,
            max_tokens=512,
            temperature=0.1
        )

        # Parse results
        parsed_results = []
        for img, result in zip(image_paths, results):
            if not result:
                logger.warning(f"Empty result for image: {img}")
                parsed_results.append({
                    "image": img,
                    "analysis": "",
                    "is_planning_map": 0
                })
                continue

            analysis, score = self._parse_result(result)
            parsed_results.append({
                "image": img,
                "analysis": analysis,
                "is_planning_map": score
            })

        # Log stats
        planning_count = sum(1 for r in parsed_results if r['is_planning_map'] == 1)
        logger.info(f"Found {planning_count}/{len(parsed_results)} planning maps")

        return parsed_results


def filter_planning_maps(image_paths: List[str],
                        batch_size: int = 500,
                        output_dir: str = None) -> List[Dict]:
    """
    Convenience function to filter planning maps

    Args:
        image_paths: List of image paths
        batch_size: Batch size
        output_dir: Output directory

    Returns:
        List of filter results
    """
    filter_obj = PlanningMapFilter()
    return filter_obj.filter(image_paths, batch_size, output_dir)
