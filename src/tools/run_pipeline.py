#!/usr/bin/env python
"""
Run the data synthesis pipeline.

Usage:
    python -m tools.run_pipeline --image_dir /path/to/images --output_dir /path/to/output
"""

import argparse
import os
import warnings

from pipeline import DataSynthesisPipeline, PipelineConfig
from tools.inference_utils import InferenceParams

warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings.*",
    category=UserWarning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PlanGPT-VL data synthesis pipeline")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--filter_images", action="store_true", help="Enable planning map filter")
    parser.add_argument("--keep_unfiltered", action="store_true", help="Keep all images after filtering")
    parser.add_argument("--response_mode", default="direct_cpt", help="Response mode for generation")
    parser.add_argument("--run_cpt", action="store_true", help="Enable CPT generation (default: on)")
    parser.add_argument("--no_cpt", action="store_true", help="Disable CPT generation")
    parser.add_argument("--run_rlaifv", action="store_true", help="Enable RLAIF-V caption refinement")
    parser.add_argument("--caption_max_iterations", type=int, default=3, help="RLAIF-V max iterations")
    parser.add_argument("--caption_max_workers", type=int, default=8, help="RLAIF-V parallel workers")
    parser.add_argument("--model", default=None, help="Model name (overrides VLLM_MODEL)")
    parser.add_argument("--server_url", default=None, help="Inference server URL")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel workers for inference")
    parser.add_argument("--empty_response_retries", type=int, default=1, help="Retries for empty responses")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    inference = InferenceParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        server_url=args.server_url or os.environ.get("VLLM_API_BASE", "http://localhost:8000"),
        model=args.model,
        max_workers=args.max_workers,
        empty_response_retries=args.empty_response_retries,
    )

    config = PipelineConfig(
        output_dir=args.output_dir,
        filter_images=args.filter_images,
        keep_unfiltered=args.keep_unfiltered,
        response_mode=args.response_mode,
        run_cpt=(args.run_cpt or not args.no_cpt),
        run_rlaifv=args.run_rlaifv,
        caption_max_iterations=args.caption_max_iterations,
        caption_max_workers=args.caption_max_workers,
        checkpoint_dir=args.checkpoint_dir,
        inference=inference,
    )
    pipeline = DataSynthesisPipeline(config)
    pipeline.run(args.image_dir)


if __name__ == "__main__":
    main()
