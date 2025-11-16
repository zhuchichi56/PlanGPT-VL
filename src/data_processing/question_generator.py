"""
Question Generator

Generates urban planning questions from images.
"""

import os
from typing import List, Dict
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.prompts import PROMPTS
from common.text_utils import parse_questions
from common.inference_utils import process_inference


class QuestionGenerator:
    """Generator for urban planning questions from images"""

    def __init__(self, prompt_key: str = "context_question_generation_od"):
        """
        Initialize question generator

        Args:
            prompt_key: Key for question generation prompt
        """
        self.prompt_key = prompt_key

    def generate(self, image_paths: List[str],
                batch_size: int = 200,
                output_dir: str = None) -> List[Dict]:
        """
        Generate questions from images

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            output_dir: Directory for checkpoints

        Returns:
            List of question dictionaries with structure:
                {
                    "image": image_path,
                    "question": question_content,
                    "type": question_type,
                    "dimension": question_dimension
                }
        """
        logger.info(f"Generating questions for {len(image_paths)} images")

        # Run inference
        results = process_inference(
            self.prompt_key,
            image_paths,
            batch_size=batch_size,
            output_dir=output_dir,
            prompts_dict=PROMPTS
        )

        # Parse questions
        all_questions = []
        for image_path, result in zip(image_paths, results):
            if not result:
                logger.warning(f"No result for {image_path}")
                continue

            parsed_questions = parse_questions(result)

            for question in parsed_questions:
                question_data = {
                    "image": image_path,
                    "question": question["content"],
                    "type": question["type"],
                    "dimension": question["dimension"]
                }
                all_questions.append(question_data)

        logger.info(f"Generated {len(all_questions)} questions")
        return all_questions


def generate_questions(image_paths: List[str],
                      batch_size: int = 200,
                      output_dir: str = None) -> List[Dict]:
    """
    Convenience function to generate questions

    Args:
        image_paths: List of image paths
        batch_size: Batch size for processing
        output_dir: Directory for checkpoints

    Returns:
        List of question dictionaries
    """
    generator = QuestionGenerator()
    return generator.generate(image_paths, batch_size, output_dir)
