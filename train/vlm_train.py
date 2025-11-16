import subprocess
import shutil
from loguru import logger
import os
import json

gpus = ["0", "1"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    with open(data_path, "r") as f:
        data = f.read()
    return data

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    print(f"Save to {data_path}")
    
def convert_format(input_file, output_file):
    """
    Convert from the original format:
    [
        {
            "image": "/path/to/image.jpeg",
            "question": "Question text?",
            "thinking": "",
            "summary": "Summary text"
        },
        ...
    ]
    
    To the target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Question text?",
                    "role": "user"
                },
                {
                    "content": "Summary text",
                    "role": "assistant"
                }
            ],
            "images": [
                "new/path/to/image.jpeg"
            ]
        },
        ...
    ]
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        # Skip if question or summary is empty
        if not item["question"].strip() or not item["summary"].strip():
            continue
            
        # Extract the image path and use it directly as input
        image_path = item["image"]
        # Use the image path as is for the output
        new_image_path = image_path
        
        # Create the new format
        new_item = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                },
                {
                    "content": item["summary"],
                    "role": "assistant"
                }
            ],
            "images": [
                new_image_path
            ]
        }
        
        output_data.append(new_item)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Output saved to {output_file}")

def convert_dpo_format(input_file, output_file):
    """
    Convert from the original format to DPO format which requires preferred and rejected responses:
    [
        {
            "image": "/path/to/image.jpeg",
            "question": "Question text?",
            "good_response": "Good response text",
            "bad_response": "Bad response text"
        },
        ...
    ]
    
    To the target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Question text?",
                    "role": "user"
                }
            ],
            "chosen": [
                {
                    "content": "Good response text",
                    "role": "assistant"
                }
            ],
            "rejected": [
                {
                    "content": "Bad response text",
                    "role": "assistant"
                }
            ],
            "images": [
                "new/path/to/image.jpeg"
            ]
        },
        ...
    ]
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        # Skip if required fields are empty
        if (not item["question"].strip() or 
            not item.get("good_response", "").strip() or 
            not item.get("bad_response", "").strip()):
            continue
            
        # Extract the image path and use it directly as input
        image_path = item["image"]
        
        # Create the new format for DPO
        new_item = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                }
            ],
            "chosen": [
                {
                    "content": item["good_response"],
                    "role": "assistant"
                }
            ],
            "rejected": [
                {
                    "content": item["bad_response"],
                    "role": "assistant"
                }
            ],
            "images": [
                image_path
            ]
        }
        
        output_data.append(new_item)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"DPO conversion complete. Output saved to {output_file}")

def run_bash_script(model_name_or_path, data_path, output_dir, template="qwen2_vl", training_type="sft"):
    logger.info(f"{training_type.upper()} training model {model_name_or_path} with template {template}")
    
    if training_type == "sft":
        convert_format(data_path, "data/selected_data_vlm.jsonl")
        dataset_name = "SELECTED_DATA_VLM"
        script_name = "ft_vl.sh"
    else:  # DPO
        convert_dpo_format(data_path, "data/selected_data_vlm_dpo.jsonl")
        dataset_name = "SELECTED_DATA_VLM_DPO"
        script_name = "ft_vl_dpo.sh"
    
    bash_command = [
        "bash", script_name,
        "--model_name_or_path", model_name_or_path,
        "--dataset", dataset_name,
        "--output_dir", output_dir,
        "--template", template,
        "--device", ",".join(gpus)
    ]
    subprocess.run(bash_command)

def sft_vl(data_paths, base_model):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/share/home/u24147/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "/share/home/u24147/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct"
    }
    ft_model_path = model_base_path[base_model]
    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/share/home/u24147/data/vlm_training/{base_model}/{name}"
        logger.info(f"Finetuning model {base_model} with data {path}")
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template="qwen2_vl",
            training_type="sft"
        )

# 最重要的参数是什么, 
def dpo_vl(data_paths, base_model, sft_model_path=None):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/share/home/u24147/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "/share/home/u24147/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct"
    }
    if sft_model_path:
        ft_model_path = sft_model_path
        logger.info(f"Using SFT model at {sft_model_path} as starting point for DPO")
    else:
        ft_model_path = model_base_path[base_model]
        logger.info(f"Using base model {base_model} directly for DPO")
    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        dpo_output_dir = f"/share/home/u24147/data/vlm_training/{base_model}/dpo_{name}"
        logger.info(f"DPO training model {base_model} with data {path}")
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=dpo_output_dir,
            template="qwen2_vl",
            training_type="dpo"
        )

if __name__ == "__main__":
    # SFT training data
    sft_data_path = [
        "/share/home/u24147/planvlm/basic_answers.json",
    ]

    
    # DPO training data (contains preferred/rejected pairs)
    dpo_data_path = [
        "/share/home/u24147/planvlm/dpo_pairs.json",
    ]
    
    # Uncomment the training type you want to run
    
    # Run SFT training
    sft_vl(sft_data_path, "qwen2-vl-7b")
    
    # Run DPO training directly on base model
    # dpo_vl(dpo_data_path, "qwen2-vl-7b")
    
    # Run DPO training on an already fine-tuned SFT model
    # sft_model = "/share/home/u24147/data/vlm_training/qwen2-vl-7b/basic_answers"
    # dpo_vl(dpo_data_path, "qwen2-vl-7b", sft_model_path=sft_model)