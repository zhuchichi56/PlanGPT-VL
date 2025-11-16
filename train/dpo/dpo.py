import subprocess
import shutil
from loguru import logger
import json

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
            
    print(f"Save to {data_path}")
    
    
def save_json(data, data_path):
    with open(data_path, "w") as f:
        json.dump(data, f)
    
    

def convert_fileA_to_fileB(fileA_data_list):

    fileB_data_list = []
    
    for fileA_data in fileA_data_list:
        fileB_data = {
            "conversations": [
                {
                    "from": "human",
                    "value": fileA_data["prompt"]
                }
            ],
            "chosen": {
                "from": "gpt",  # Using "gpt" as this appears to be the convention in fileB
                "value": fileA_data["chosen"][1]["content"] if len(fileA_data["chosen"]) > 1 else ""
            },
            "rejected": {
                "from": "gpt",
                "value": fileA_data["rejected"][1]["content"] if len(fileA_data["rejected"]) > 1 else ""
            }
        }
        fileB_data_list.append(fileB_data)
    
    return fileB_data_list


    
def run_bash_script(model_name_or_path, data_path, output_dir, template="mistral"):
    logger.info(f"DPO training model {model_name_or_path} with template {template}")
    
    fileA_data_list = load_json(data_path)
    fileB_data_list = convert_fileA_to_fileB(fileA_data_list)
    save_json(fileB_data_list, "/HOME/sustc_ghchen/sustc_ghchen_4/LLaMA-Factory/data/selected_data_dpo.json")
    # shutil.copy(data_path, "/HOME/sustc_ghchen/sustc_ghchen_4/LLaMA-Factory/data/selected_data_dpo.json")
    bash_command = [
        "bash", "dpo.sh",
        "--model_name_or_path", model_name_or_path,
        "--dataset", "SELECTED_DATA_DPO", 
        "--output_dir", output_dir,
        "--template", template,
        "--device", "0,1",
    ]
    subprocess.run(bash_command)


def select_template(base_model):
    template_match = {
        "qwen25-7B-Instruct": "qwen",
        "llama2-7b": "llama3",
        "llama3-8b": "llama3",
        "llama3.1-8b": "llama2",
        "llama3.2-3b": "llama2",
        "mistralv3-8b": "mistral",
        "llama3-8b-instruct": "llama3"
    }
    return template_match[base_model]


def dpo(data_paths, base_model):
    assert base_model in ["llama2-7b", "llama3-8b", "llama3.1-8b", "mistralv3-7b", "llama3.2-3b", "mistralv3-8b", "qwen25-7B-Instruct", "llama3-8b-instruct"]
    model_base_path = {
        "qwen25-7B-Instruct": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2.5-7B-instruct",
        "llama3-8b-instruct":"/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/Meta-Llama-3-8B-Instruct",
    }
    
    
    ft_model_path = model_base_path[base_model]
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/dpo/{base_model}-{name}"
        logger.info(f"DPO training model {base_model} with data {path}")
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template=select_template(base_model)
        )
        
if __name__ == "__main__":
    data_path = [
        "/HOME/sustc_ghchen/sustc_ghchen_4/dpo_selection/ultrafeedback_binarized_10.json",
        "/HOME/sustc_ghchen/sustc_ghchen_4/dpo_selection/ultrafeedback_binarized_25.json",
        "/HOME/sustc_ghchen/sustc_ghchen_4/dpo_selection/ultrafeedback_binarized_50.json",
        "/HOME/sustc_ghchen/sustc_ghchen_4/dpo_selection/ultrafeedback_binarized.json"
    ]

    for path in data_path:
        dpo([path], "llama3-8b-instruct")
        
    

