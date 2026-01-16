import argparse
from pathlib import Path

from loguru import logger

from ft_tuning_vlm_simple import convert_sft, run_sft, update_dataset_info


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of VLM SFT jobs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--device", default="0")
    parser.add_argument("--question_key", default="question")
    parser.add_argument("--response_key", default="response")
    parser.add_argument("--image_key", default="image")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        stem = Path(input_path).stem
        dataset_name = f"{stem}_sft"
        output_file = f"{dataset_name}.json"
        output_path = str(DATA_DIR / output_file)

        logger.info("Preparing dataset {} from {}", dataset_name, input_path)
        convert_sft(input_path, output_path, args.question_key, args.response_key, args.image_key)
        update_dataset_info(dataset_name, output_file, ranking=False)

        output_dir = str(output_root / stem)
        run_sft(dataset_name, args.model_name_or_path, output_dir, args.template, args.device)


if __name__ == "__main__":
    main()
