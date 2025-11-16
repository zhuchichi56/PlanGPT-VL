"""
Inference Utilities

Provides batch inference capabilities with checkpointing support.
"""

import os
from typing import List, Dict, Optional
from loguru import logger

# Import from inference client
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.client import parallel_image_inference

from .io_utils import load_json, save_json


def parallel_image_inference_batch(
    prompt_list: List[str],
    image_paths: List[str],
    batch_size: int = 200,
    output_dir: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    server_urls: Optional[List[str]] = None
) -> List[str]:
    """
    Batch inference with checkpoint support

    Args:
        prompt_list: List of prompts
        image_paths: List of image paths
        batch_size: Batch size for processing
        output_dir: Directory for checkpoints
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        server_urls: List of server URLs for load balancing

    Returns:
        List of inference results
    """
    assert len(prompt_list) == len(image_paths), \
        f"prompt_list length ({len(prompt_list)}) != image_paths length ({len(image_paths)})"

    # Setup checkpoint
    checkpoint_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_file = os.path.join(output_dir, "inference_checkpoint.json")

    # Load checkpoint if exists
    completed_indices = set()
    all_results = [None] * len(image_paths)

    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_json(checkpoint_file)
            completed_indices = set(checkpoint_data["completed_indices"])

            for idx, result in checkpoint_data["results"].items():
                idx = int(idx)
                if idx < len(all_results):
                    all_results[idx] = result

            logger.info(f"Resumed from checkpoint: {len(completed_indices)} results")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # Process in batches
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        # Get pending indices
        batch_indices = list(range(start_idx, end_idx))
        pending_indices = [idx for idx in batch_indices if idx not in completed_indices]

        if not pending_indices:
            logger.info(f"Batch {batch_idx+1}/{total_batches} already processed, skipping")
            continue

        logger.info(f"Processing batch {batch_idx+1}/{total_batches}, {len(pending_indices)} items")

        # Prepare batch data
        batch_image_paths = [image_paths[idx] for idx in pending_indices]
        batch_prompts = [prompt_list[idx] for idx in pending_indices]

        try:
            # Run inference
            batch_results = parallel_image_inference(
                batch_prompts,
                batch_image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                server_urls=server_urls
            )

            # Update results
            for i, idx in enumerate(pending_indices):
                all_results[idx] = batch_results[i]
                completed_indices.add(idx)

            # Save checkpoint
            if checkpoint_file:
                results_dict = {str(i): res for i, res in enumerate(all_results) if res is not None}
                checkpoint_data = {
                    "completed_indices": list(completed_indices),
                    "results": results_dict
                }
                save_json(checkpoint_data, checkpoint_file)
                logger.info(f"Checkpoint saved: {len(completed_indices)}/{len(image_paths)} completed")

        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed: {e}")
            # Continue to next batch on error

    # Check completion
    if len(completed_indices) < len(image_paths):
        logger.warning(f"{len(image_paths) - len(completed_indices)} items not completed")

    return all_results


def process_inference(
    prompt_template: str,
    image_paths: List[str],
    params: Optional[List[Dict]] = None,
    batch_size: int = 200,
    output_dir: Optional[str] = None,
    prompts_dict: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Process inference with prompt template

    Args:
        prompt_template: Template key or template string
        image_paths: List of image paths
        params: List of parameters for formatting prompts
        batch_size: Batch size
        output_dir: Output directory for checkpoints
        prompts_dict: Dictionary of prompt templates

    Returns:
        List of inference results
    """
    if not image_paths:
        return []

    # Setup checkpoint
    checkpoint_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_file = os.path.join(output_dir, f"{prompt_template}_checkpoint.json")

    # Get template
    if prompts_dict:
        template = prompts_dict.get(prompt_template, prompt_template)
    else:
        template = prompt_template

    # Format prompts
    if params:
        prompts = [template.format(**p) for p in params]
    else:
        prompts = [template] * len(image_paths)

    # Load checkpoint if exists
    completed_indices = set()
    all_results = [None] * len(image_paths)

    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_json(checkpoint_file)
            completed_indices = set(checkpoint_data["completed_indices"])

            for idx, result in checkpoint_data["results"].items():
                idx = int(idx)
                if idx < len(all_results):
                    all_results[idx] = result

            logger.info(f"Resumed from checkpoint: {len(completed_indices)} results")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # Process in batches
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        batch_indices = list(range(start_idx, end_idx))
        pending_indices = [idx for idx in batch_indices if idx not in completed_indices]

        if not pending_indices:
            logger.info(f"Batch {batch_idx+1}/{total_batches} already processed, skipping")
            continue

        logger.info(f"Processing batch {batch_idx+1}/{total_batches}, {len(pending_indices)} items")

        batch_image_paths = [image_paths[idx] for idx in pending_indices]
        batch_prompts = [prompts[idx] for idx in pending_indices]

        try:
            batch_results = parallel_image_inference(
                batch_prompts,
                batch_image_paths,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9
            )

            for i, idx in enumerate(pending_indices):
                all_results[idx] = batch_results[i]
                completed_indices.add(idx)

            if checkpoint_file:
                results_dict = {str(i): res for i, res in enumerate(all_results) if res is not None}
                checkpoint_data = {
                    "completed_indices": list(completed_indices),
                    "results": results_dict
                }
                save_json(checkpoint_data, checkpoint_file)
                logger.info(f"Checkpoint saved: {len(completed_indices)}/{len(image_paths)} completed")

        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed: {e}")

    if len(completed_indices) < len(image_paths):
        logger.warning(f"{len(image_paths) - len(completed_indices)} items not completed")

    return all_results
