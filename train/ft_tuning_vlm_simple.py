import argparse
import json
import os
import subprocess
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_INFO = DATA_DIR / "dataset_info.json"


def load_json(data_path: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], data_path: str) -> None:
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_dataset_info(dataset_name: str, file_name: str, ranking: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_INFO.exists():
        with open(DATA_INFO, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    entry = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    if ranking:
        entry["ranking"] = True
        entry["columns"].update({"chosen": "chosen", "rejected": "rejected"})

    dataset_info[dataset_name] = entry

    with open(DATA_INFO, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def convert_sft(input_file: str, output_file: str, question_key: str, response_key: str, image_key: str) -> None:
    input_data = load_json(input_file)
    output_data = []

    for item in input_data:
        question = str(item.get(question_key, "")).strip()
        response = str(item.get(response_key, "")).strip()
        if not question or not response:
            continue

        image_path = item.get(image_key, "")
        messages = [
            {"content": f"<image>{question}", "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        images = [image_path] if image_path else []
        output_data.append({"messages": messages, "images": images})

    save_json(output_data, output_file)


def convert_dpo(
    input_file: str,
    output_file: str,
    question_key: str,
    good_key: str,
    bad_key: str,
    image_key: str,
) -> None:
    input_data = load_json(input_file)
    output_data = []

    for item in input_data:
        question = str(item.get(question_key, "")).strip()
        good_resp = str(item.get(good_key, "")).strip()
        bad_resp = str(item.get(bad_key, "")).strip()
        if not question or not good_resp or not bad_resp:
            continue

        image_path = item.get(image_key, "")
        messages = [{"content": f"<image>{question}", "role": "user"}]
        chosen = {"content": good_resp, "role": "assistant"}
        rejected = {"content": bad_resp, "role": "assistant"}
        images = [image_path] if image_path else []
        output_data.append(
            {
                "messages": messages,
                "chosen": chosen,
                "rejected": rejected,
                "images": images,
            }
        )

    save_json(output_data, output_file)


def run_sft(
    dataset_name: str,
    model_name_or_path: str,
    output_dir: str,
    template: str,
    device: str,
) -> None:
    cmd = [
        "bash",
        str(SCRIPT_DIR / "ft_vl.sh"),
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


def run_dpo(
    dataset_name: str,
    model_name_or_path: str,
    output_dir: str,
    template: str,
    device: str,
    finetuning_type: str | None,
    freeze_vision_tower: str | None,
    freeze_multi_modal_projector: str | None,
    freeze_language_model: str | None,
    global_batch_size: str | None,
) -> None:
    cmd = [
        "bash",
        str(SCRIPT_DIR / "dpo_vl.sh"),
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
    if finetuning_type:
        cmd += ["--finetuning_type", finetuning_type]
    if freeze_vision_tower:
        cmd += ["--freeze_vision_tower", freeze_vision_tower]
    if freeze_multi_modal_projector:
        cmd += ["--freeze_multi_modal_projector", freeze_multi_modal_projector]
    if freeze_language_model:
        cmd += ["--freeze_language_model", freeze_language_model]
    if global_batch_size:
        cmd += ["--global_batch_size", global_batch_size]

    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM SFT/DPO entrypoint")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    sft_parser = subparsers.add_parser("sft", help="Run SFT with local JSON data")
    sft_parser.add_argument("--input", required=True, help="Input JSON file")
    sft_parser.add_argument("--output_dir", required=True, help="SFT output directory")
    sft_parser.add_argument("--model_name_or_path", required=True)
    sft_parser.add_argument("--template", default="qwen2_vl")
    sft_parser.add_argument("--device", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    sft_parser.add_argument("--dataset_name", default="selected_data_vlm")
    sft_parser.add_argument("--question_key", default="question")
    sft_parser.add_argument("--response_key", default="response")
    sft_parser.add_argument("--image_key", default="image")

    dpo_parser = subparsers.add_parser("dpo", help="Run DPO with local JSON data")
    dpo_parser.add_argument("--input", required=True, help="Input JSON file")
    dpo_parser.add_argument("--output_dir", required=True, help="DPO output directory")
    dpo_parser.add_argument("--model_name_or_path", required=True)
    dpo_parser.add_argument("--template", default="qwen2_vl")
    dpo_parser.add_argument("--device", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    dpo_parser.add_argument("--dataset_name", default="selected_data_vlm_dpo")
    dpo_parser.add_argument("--question_key", default="question")
    dpo_parser.add_argument("--good_key", default="good_response")
    dpo_parser.add_argument("--bad_key", default="bad_response")
    dpo_parser.add_argument("--image_key", default="image")
    dpo_parser.add_argument("--finetuning_type")
    dpo_parser.add_argument("--freeze_vision_tower")
    dpo_parser.add_argument("--freeze_multi_modal_projector")
    dpo_parser.add_argument("--freeze_language_model")
    dpo_parser.add_argument("--global_batch_size")

    args = parser.parse_args()

    if args.stage == "sft":
        output_file = f"{args.dataset_name}.json"
        output_path = str(DATA_DIR / output_file)
        convert_sft(args.input, output_path, args.question_key, args.response_key, args.image_key)
        update_dataset_info(args.dataset_name, output_file, ranking=False)
        logger.info("SFT dataset ready at {}", output_path)
        run_sft(args.dataset_name, args.model_name_or_path, args.output_dir, args.template, args.device)
    else:
        output_file = f"{args.dataset_name}.json"
        output_path = str(DATA_DIR / output_file)
        convert_dpo(
            args.input,
            output_path,
            args.question_key,
            args.good_key,
            args.bad_key,
            args.image_key,
        )
        update_dataset_info(args.dataset_name, output_file, ranking=True)
        logger.info("DPO dataset ready at {}", output_path)
        run_dpo(
            args.dataset_name,
            args.model_name_or_path,
            args.output_dir,
            args.template,
            args.device,
            args.finetuning_type,
            args.freeze_vision_tower,
            args.freeze_multi_modal_projector,
            args.freeze_language_model,
            args.global_batch_size,
        )


if __name__ == "__main__":
    main()
