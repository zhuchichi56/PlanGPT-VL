import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_INFO = DATA_DIR / "dataset_info.json"


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_json(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


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


def convert_alpaca(input_file: str, output_file: str) -> None:
    data = load_jsonl(input_file)
    output_data = []
    for line in data:
        instruction = line.get("instruction", "")
        inp = line.get("input", "")
        response = line.get("output", "")
        if inp:
            instruction = f"{instruction}\n{inp}"
        output_data.append({"instruction": instruction, "response": response})
    save_jsonl(output_data, output_file)


def check_images(input_file: str, image_key: str) -> list[str]:
    data = load_json(input_file)
    missing = []
    for item in data:
        image_path = item.get(image_key, "")
        if image_path and not Path(image_path).exists():
            missing.append(image_path)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset tools")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    sft_parser = subparsers.add_parser("sft")
    sft_parser.add_argument("--input", required=True)
    sft_parser.add_argument("--output", required=True)
    sft_parser.add_argument("--dataset_name")
    sft_parser.add_argument("--question_key", default="question")
    sft_parser.add_argument("--response_key", default="response")
    sft_parser.add_argument("--image_key", default="image")

    dpo_parser = subparsers.add_parser("dpo")
    dpo_parser.add_argument("--input", required=True)
    dpo_parser.add_argument("--output", required=True)
    dpo_parser.add_argument("--dataset_name")
    dpo_parser.add_argument("--question_key", default="question")
    dpo_parser.add_argument("--good_key", default="good_response")
    dpo_parser.add_argument("--bad_key", default="bad_response")
    dpo_parser.add_argument("--image_key", default="image")

    alpaca_parser = subparsers.add_parser("alpaca")
    alpaca_parser.add_argument("--input", required=True)
    alpaca_parser.add_argument("--output", required=True)
    alpaca_parser.add_argument("--dataset_name")

    check_parser = subparsers.add_parser("check_images")
    check_parser.add_argument("--input", required=True)
    check_parser.add_argument("--image_key", default="image")

    args = parser.parse_args()

    if args.cmd == "sft":
        convert_sft(args.input, args.output, args.question_key, args.response_key, args.image_key)
        if args.dataset_name:
            update_dataset_info(args.dataset_name, Path(args.output).name, ranking=False)
    elif args.cmd == "dpo":
        convert_dpo(args.input, args.output, args.question_key, args.good_key, args.bad_key, args.image_key)
        if args.dataset_name:
            update_dataset_info(args.dataset_name, Path(args.output).name, ranking=True)
    elif args.cmd == "alpaca":
        convert_alpaca(args.input, args.output)
        if args.dataset_name:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if DATA_INFO.exists():
                with open(DATA_INFO, "r", encoding="utf-8") as f:
                    dataset_info = json.load(f)
            else:
                dataset_info = {}
            dataset_info[args.dataset_name] = {
                "file_name": Path(args.output).name,
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "response": "response"},
            }
            with open(DATA_INFO, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    else:
        missing = check_images(args.input, args.image_key)
        if missing:
            print("Missing images:")
            for path in missing:
                print(path)
        else:
            print("All images exist.")


if __name__ == "__main__":
    main()
