import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def run_sft(args: argparse.Namespace) -> None:
    cmd = [
        "bash",
        str(SCRIPT_DIR / "ft_vl.sh"),
        "--model_name_or_path",
        args.sft_model_name_or_path,
        "--dataset",
        args.sft_dataset,
        "--output_dir",
        args.sft_output_dir,
        "--template",
        args.template,
        "--device",
        args.device,
    ]
    run_cmd(cmd)


def run_dpo(args: argparse.Namespace) -> None:
    cmd = [
        "bash",
        str(SCRIPT_DIR / "dpo_vl.sh"),
        "--model_name_or_path",
        args.dpo_model_name_or_path,
        "--dataset",
        args.dpo_dataset,
        "--output_dir",
        args.dpo_output_dir,
        "--template",
        args.template,
        "--device",
        args.device,
    ]
    if args.dpo_finetuning_type:
        cmd += ["--finetuning_type", args.dpo_finetuning_type]
    if args.freeze_vision_tower:
        cmd += ["--freeze_vision_tower", args.freeze_vision_tower]
    if args.freeze_multi_modal_projector:
        cmd += ["--freeze_multi_modal_projector", args.freeze_multi_modal_projector]
    if args.freeze_language_model:
        cmd += ["--freeze_language_model", args.freeze_language_model]
    if args.global_batch_size:
        cmd += ["--global_batch_size", args.global_batch_size]

    run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT then DPO pipeline")
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--device", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))

    parser.add_argument("--sft_dataset", required=True)
    parser.add_argument("--sft_model_name_or_path", required=True)
    parser.add_argument("--sft_output_dir", required=True)

    parser.add_argument("--dpo_dataset", required=True)
    parser.add_argument("--dpo_model_name_or_path", required=True)
    parser.add_argument("--dpo_output_dir", required=True)
    parser.add_argument("--dpo_finetuning_type")
    parser.add_argument("--freeze_vision_tower")
    parser.add_argument("--freeze_multi_modal_projector")
    parser.add_argument("--freeze_language_model")
    parser.add_argument("--global_batch_size")

    args = parser.parse_args()

    run_sft(args)
    run_dpo(args)


if __name__ == "__main__":
    main()
