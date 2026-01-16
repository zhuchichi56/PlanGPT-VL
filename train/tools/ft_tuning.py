import argparse
import json
import subprocess
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_INFO = DATA_DIR / "dataset_info.json"


def load_jsonl(data_path: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: list[dict], data_path: str) -> None:
    with open(data_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def convert_instruction_input_output(data_path: str) -> list[dict]:
    data = load_jsonl(data_path)
    new_data = []
    for line in data:
        instruction = line.get("instruction", "")
        inp = line.get("input", "")
        response = line.get("output", "")
        if inp:
            instruction = f"{instruction}\n{inp}"
        new_data.append({"instruction": instruction, "response": response})
    return new_data


def update_dataset_info(dataset_name: str, file_name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_INFO.exists():
        with open(DATA_INFO, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "response": "response",
        },
    }

    with open(DATA_INFO, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def run_sft(model_name_or_path: str, dataset_name: str, output_dir: str, template: str, device: str) -> None:
    cmd = [
        "bash",
        str(SCRIPT_DIR / "dpo" / "ft.sh"),
        "--model_name_or_path",
        model_name_or_path,
        "--dataset",
        dataset_name,
        "--output_dir",
        output_dir,
        "--template",
        template,
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Text SFT entrypoint")
    parser.add_argument("--input", required=True, help="Input JSONL with instruction/input/output")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--template", default="alpaca")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dataset_name", default="selected_data")
    args = parser.parse_args()

    output_file = f"{args.dataset_name}.jsonl"
    output_path = str(DATA_DIR / output_file)

    logger.info("Preparing dataset {}", args.dataset_name)
    converted = convert_instruction_input_output(args.input)
    save_jsonl(converted, output_path)
    update_dataset_info(args.dataset_name, output_file)

    run_sft(args.model_name_or_path, args.dataset_name, args.output_dir, args.template, args.device)


if __name__ == "__main__":
    main()
