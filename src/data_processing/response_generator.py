"""
Response Generator

Generates responses for urban planning questions.
"""

import os
from typing import List, Dict
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.prompts import PROMPTS
from common.inference_utils import process_inference


class ResponseGenerator:
    """Generator for urban planning question responses"""

    def __init__(self):
        pass

    def generate(self,
                question_results: List[Dict],
                mode: str = "direct_cpt",
                question_key: str = "question",
                caption_key: str = "caption",
                batch_size: int = 200,
                output_dir: str = None) -> List[str]:
        """
        Generate responses for questions

        Args:
            question_results: List of dictionaries containing 'image' and question field
            mode: Generation mode ('direct', 'with_caption', 'direct_cpt', 'with_caption_cpt')
            question_key: Key for question field in input data
            caption_key: Key for caption field (if using caption mode)
            batch_size: Batch size for processing
            output_dir: Directory for checkpoints

        Returns:
            List of generated responses
        """
        # Select prompt template based on mode
        template_map = {
            "direct": "answer_direct",
            "with_caption": "answer_with_caption",
            "direct_cpt": "answer_direct_cpt",
            "with_caption_cpt": "answer_with_caption_cpt",
            "cpt": "answer_direct_cpt"  # Alias
        }

        template_key = template_map.get(mode, "answer_direct_cpt")
        logger.info(f"Using prompt template: {template_key}")

        # Prepare image paths and parameters
        image_paths = [item["image"] for item in question_results]

        # Prepare parameters for prompt formatting
        params = []
        for item in question_results:
            param = {"question": item[question_key]}
            if "caption" in template_key and caption_key in item:
                param["caption"] = item[caption_key]
            params.append(param)

        # Run inference
        results = process_inference(
            template_key,
            image_paths,
            params=params,
            batch_size=batch_size,
            output_dir=output_dir,
            prompts_dict=PROMPTS
        )

        return results


def generate_responses(question_results: List[Dict],
                      mode: str = "direct_cpt",
                      batch_size: int = 200,
                      output_dir: str = None) -> List[str]:
    """
    Convenience function to generate responses

    Args:
        question_results: List of question dictionaries
        mode: Generation mode
        batch_size: Batch size
        output_dir: Checkpoint directory

    Returns:
        List of responses
    """
    generator = ResponseGenerator()
    return generator.generate(question_results, mode, batch_size=batch_size, output_dir=output_dir)
