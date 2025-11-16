import subprocess
import shutil
from loguru import logger
import os
import json

gpus = ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)  # Use json.load instead of f.read()
    return data

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    
    print(f"Save to {data_path}")

# instruction, input, output -> instruction, response
def convert_data(data_path):
    data = load_json(data_path)  
    new_data = [{"instruction": item["instruction"] if item["input"] == "" else item["instruction"] + "\n" + item["input"], 
                 "response": item["output"]} for item in data]
    save_jsonl(new_data, data_path + "_converted.jsonl")
    return data_path + "_converted.jsonl"

# 中文
def run_bash_script(model_name_or_path, data_path, output_dir, template="qwen"):
    logger.info(f"Finetuning model {model_name_or_path} with template {template}")
    # Uncomment if you need to convert data
    data_path = convert_data(data_path)
    shutil.copy(data_path, "data/selected_data.jsonl")
    bash_command = [
        "bash", "ft.sh",
        "--model_name_or_path", model_name_or_path,
        "--dataset", "SELECTED_DATA",
        "--output_dir", output_dir,
        "--template", template,
        "--device", ",".join(gpus)
    ]
    subprocess.run(bash_command)

def sft(data_paths, base_model):
    # Make sure all models listed in the assert are also in the dictionary
    model_base_path = {
        "llama2-7b": "/path/to/llama2-7b",  # Add path for this model
        "llama3-8b": "/share/home/tj24147/data/huggingface_model/LLaMA/Meta-Llama-3-8B",
        "llama3.1-8b": "/home/zhe/models/Meta-Llama-3.1-8B",
        "llama3.2-3b": "/data/zhe/models/llama3.2-3b",
        "mistralv3-7b": "/path/to/mistralv3-7b",  # Add path for this model
        "qwen2-7b": "/share/home/tj24147/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B"
    }
    
    assert base_model in model_base_path, f"Model {base_model} not found in model_base_path"
    
    ft_model_path = model_base_path[base_model]
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/share/home/tj24147/data/new/{base_model}-{name}"
        logger.info(f"Finetuning model {base_model} with data {path}")
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir
        )
        

if __name__ == "__main__":
    data_path =[
        "/share/home/tj24147/PKU-UPGLM/urban_planning_model/data/plangpt_demo.json"
    ]

    for path in data_path:
        # 3b
        sft([path], "qwen2-7b")
        
        
        

        
    
        
        
        