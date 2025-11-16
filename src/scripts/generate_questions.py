#!/usr/bin/env python
"""
Question Generation Script

Generates urban planning questions from images.

Usage:
    python -m scripts.generate_questions --image_dir /path/to/images --output /path/to/output.json
"""

import os
import sys
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.image_utils import process_image_directory
from common.io_utils import save_json
from data_processing.question_generator import generate_questions


def main(image_dir: str,
         output_path: str,
         batch_size: int = 200,
         checkpoint_dir: str = None):
    """
    Generate questions from images

    Args:
        image_dir: Directory containing images
        output_path: Path to save generated questions (JSON)
        batch_size: Batch size for processing (default: 200)
        checkpoint_dir: Directory for saving checkpoints (optional)
    """
    # Get all images
    image_paths = process_image_directory(image_dir)
    print(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("No images found!")
        return

    # Shuffle for diversity
    import random
    random.shuffle(image_paths)

    # Generate questions
    questions = generate_questions(
        image_paths,
        batch_size=batch_size,
        output_dir=checkpoint_dir
    )

    # Save results
    save_json(questions, output_path)
    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
