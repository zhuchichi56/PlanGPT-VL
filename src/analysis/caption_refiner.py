"""
Caption Refiner for RLAIF-V

Iteratively refines image captions to improve quality.
Based on the RLAIF-V framework: https://github.com/RLHF-V/RLAIF-V
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.prompts import PROMPTS
from data_processing.response_generator import ResponseGenerator
from common.io_utils import load_json, save_json


class CaptionRefiner:
    """Refines image captions iteratively"""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize caption refiner

        Args:
            api_key: API key for evaluator (if using external evaluation)
            api_base: API base URL for evaluator
        """
        self.api_key = api_key
        self.api_base = api_base
        self.response_gen = ResponseGenerator()

    def parse_key_points(self, caption: str) -> List[str]:
        """
        Parse numbered key points from caption

        Args:
            caption: Caption text with [1], [2], etc.

        Returns:
            List of key points
        """
        key_points = []
        pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, caption, re.DOTALL)

        for _, content in matches:
            key_point = content.strip()
            if key_point:
                key_points.append(key_point)

        # Fallback: split by paragraphs
        if not key_points:
            paragraphs = [p.strip() for p in caption.split('\n\n') if p.strip()]
            if paragraphs:
                key_points = paragraphs

        return key_points

    def generate_caption(self, item: Dict) -> str:
        """
        Generate initial caption for an item

        Args:
            item: Dictionary with 'image' and optionally 'question'

        Returns:
            Generated caption
        """
        temp_item = {
            "image": item["image"],
            "question": PROMPTS["caption_generation"]
        }

        results = self.response_gen.generate(
            [temp_item],
            mode="direct",
            question_key="question"
        )

        return results[0] if results else ""

    def improve_key_point(self, item: Dict, key_point: str, index: int) -> str:
        """
        Improve a single key point

        Args:
            item: Item dictionary with 'image' and 'question'
            key_point: Current key point text
            index: Index of the key point

        Returns:
            Improved key point or empty string if no improvement needed
        """
        prompt = PROMPTS["caption_improvement"].format(
            question=item.get("question", ""),
            index=index,
            key_point=key_point
        )

        temp_item = {
            "image": item["image"],
            "question": prompt
        }

        results = self.response_gen.generate(
            [temp_item],
            mode="direct",
            question_key="question"
        )

        result = results[0] if results else ""

        if "无需改进" in result:
            return ""

        # Extract improved version
        improved_point = ""
        patterns = [
            r'改进版本：\s*(.*?)$',
            r'改进的描述点：\s*(.*?)$',
            r'建议修改为：\s*(.*?)$'
        ]

        for pattern in patterns:
            matches = re.search(pattern, result, re.DOTALL)
            if matches:
                improved_point = matches.group(1).strip()
                break

        # Fallback: use lines after keywords
        if not improved_point:
            lines = result.strip().split('\n')
            for i, line in enumerate(lines):
                if "改进" in line or "建议" in line:
                    if i+1 < len(lines):
                        improved_point = lines[i+1].strip()
                        break

        return improved_point

    def improve_caption(self, item: Dict, current_caption: str) -> str:
        """
        Improve entire caption by refining each key point

        Args:
            item: Item dictionary
            current_caption: Current caption text

        Returns:
            Improved caption
        """
        key_points = self.parse_key_points(current_caption)
        improved_points = []

        for i, point in enumerate(key_points):
            improved = self.improve_key_point(item, point, i+1)
            if improved:
                improved_points.append(f"[{i+1}] {improved}")
            else:
                improved_points.append(f"[{i+1}] {point}")

        return "\n\n".join(improved_points)

    def iterate_caption(self,
                       item: Dict,
                       max_iterations: int = 3) -> Tuple[str, List[str]]:
        """
        Iteratively improve caption

        Args:
            item: Item dictionary with 'image' and optionally 'question'
            max_iterations: Maximum improvement iterations

        Returns:
            Tuple of (best_caption, all_captions)
        """
        current_caption = self.generate_caption(item)
        all_captions = [current_caption]

        for i in range(max_iterations - 1):
            print(f"Iteration {i+1}/{max_iterations-1} for image {os.path.basename(item['image'])}")
            improved_caption = self.improve_caption(item, current_caption)
            all_captions.append(improved_caption)
            current_caption = improved_caption

        # Return last caption as best (in simple version without scoring)
        return current_caption, all_captions

    def process_dataset(self,
                       data: List[Dict],
                       max_iterations: int = 3,
                       max_workers: int = 8) -> List[Dict]:
        """
        Process entire dataset with caption refinement

        Args:
            data: List of item dictionaries
            max_iterations: Maximum iterations per caption
            max_workers: Maximum parallel workers

        Returns:
            Updated data with refined captions
        """
        def process_item(item):
            try:
                best_caption, all_captions = self.iterate_caption(item, max_iterations)

                result = item.copy()
                result["caption"] = best_caption
                result["all_captions"] = all_captions

                print(f"Processed {os.path.basename(item['image'])}")
                return result

            except Exception as e:
                print(f"Error processing item: {str(e)}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [executor.submit(process_item, item) for item in data]

            for future in tqdm(tasks, desc="Processing images"):
                result = future.result()
                if result:
                    results.append(result)

        print(f"Processed {len(results)} items")
        return results


def refine_captions(data_path: str,
                   output_path: str,
                   max_iterations: int = 3,
                   max_workers: int = 8) -> List[Dict]:
    """
    Convenience function to refine captions

    Args:
        data_path: Path to input data JSON
        output_path: Path to output JSON
        max_iterations: Maximum iterations per caption
        max_workers: Maximum parallel workers

    Returns:
        List of results with refined captions
    """
    data = load_json(data_path)
    refiner = CaptionRefiner()
    results = refiner.process_dataset(data, max_iterations, max_workers)
    save_json(results, output_path)
    print(f"Results saved to {output_path}")
    return results
