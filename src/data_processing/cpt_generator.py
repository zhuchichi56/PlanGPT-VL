"""
Critical Point Thinking (CPT) Generator

Extracts critical thinking points from question-answer pairs.
"""

import os
from typing import List, Dict
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.prompts import PROMPTS
from common.text_utils import extract_thinking_content, combine_thinking_answer
from common.inference_utils import process_inference


class CPTGenerator:
    """Generator for critical point thinking"""

    def __init__(self, version: str = "critical_version"):
        """
        Initialize CPT generator

        Args:
            version: CPT version to use
        """
        self.version = version

    def generate(self,
                data: List[Dict],
                question_key: str = "question",
                answer_key: str = "answer",
                batch_size: int = 200,
                output_dir: str = None) -> List[Dict]:
        """
        Generate critical thinking points

        Args:
            data: List of dictionaries with 'image', question and answer
            question_key: Key for question field
            answer_key: Key for answer field
            batch_size: Batch size
            output_dir: Checkpoint directory

        Returns:
            Updated data with CPT responses
        """
        logger.info(f"Generating CPT for {len(data)} items")

        # Prepare data
        image_paths = [item["image"] for item in data]
        params = [
            {"question": item[question_key], "answer": item[answer_key]}
            for item in data
        ]

        # Run inference
        results = process_inference(
            self.version,
            image_paths,
            params=params,
            batch_size=batch_size,
            output_dir=output_dir,
            prompts_dict=PROMPTS
        )

        # Process results
        for result, item in zip(results, data):
            if not result:
                logger.warning(f"No CPT result for item")
                continue

            thinking = extract_thinking_content(result)
            item[f"{self.version}_response"] = combine_thinking_answer(thinking, item[answer_key])

        return data


def generate_cpt(data: List[Dict],
                version: str = "critical_version",
                batch_size: int = 200,
                output_dir: str = None) -> List[Dict]:
    """
    Convenience function to generate CPT

    Args:
        data: List of question-answer dictionaries
        version: CPT version
        batch_size: Batch size
        output_dir: Checkpoint directory

    Returns:
        Updated data with CPT
    """
    generator = CPTGenerator(version)
    return generator.generate(data, batch_size=batch_size, output_dir=output_dir)
