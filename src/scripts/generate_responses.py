#!/usr/bin/env python
"""
Response Generation Script

Generates responses for urban planning questions.

Usage:
    python -m scripts.generate_responses --input questions.json --output responses.json --mode direct_cpt
"""

import os
import sys
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.io_utils import load_json, save_json
from data_processing.response_generator import generate_responses


def main(input_path: str,
         output_path: str,
         mode: str = "direct_cpt",
         batch_size: int = 200,
         checkpoint_dir: str = None):
    """
    Generate responses for questions

    Args:
        input_path: Path to questions JSON
        output_path: Path to save responses
        mode: Response mode ('direct', 'with_caption', 'direct_cpt', 'with_caption_cpt')
        batch_size: Batch size (default: 200)
        checkpoint_dir: Checkpoint directory (optional)
    """
    # Load questions
    data = load_json(input_path)
    print(f"Loaded {len(data)} questions from {input_path}")

    # Generate responses
    responses = generate_responses(
        data,
        mode=mode,
        batch_size=batch_size,
        output_dir=checkpoint_dir
    )

    # Add responses to data
    for item, response in zip(data, responses):
        item["response"] = response

    # Save
    save_json(data, output_path)
    print(f"Saved responses to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
