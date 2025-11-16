import subprocess
from loguru import logger
import os
import json
import json
import os
from collections import defaultdict
import random
random.seed(42)

gpus = ["0", "1"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def load_jsonl(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data
def save_json(data, data_path):
    with open(data_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Save to {data_path}")

def save_jsonl(data, data_path):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    print(f"Save to {data_path}")
    
    



def covert_caption_to_basic_answer(input_file, output_file):
    data = load_json(input_file)

    import random
    output_data = []    
    question = [
        "作为一个城市规划师，请仔细描述这张规划图。",
        "请描述这张规划图。",
        "请详细描述这张规划图。",
        "作为一名城市规划师，请详细描述这张规划图。",
        "作为一名城市规划师，请仔细描述这张规划图。"
    ]
    random.shuffle(question)
    for item in data:
        new_item = {
            "image": item["image"],
            "question": random.choice(question),
            "basic_answer": item["caption"]
        }
        output_data.append(new_item)
    
    save_json(output_data, output_file)
    
    
def convert_format(input_file, output_file):
    """
    Convert from the original format to the target format with multiple messages and images,
    grouping conversations by image.
    
    Original format example:
    [
        {
            "image": "/path/to/image1.jpeg",
            "question": "Who are they?",
            "basic_answer": "They're Kane and Gretzka from Bayern Munich."
        },
        {
            "image": "/path/to/image1.jpeg",
            "question": "What are they doing?",
            "basic_answer": "They are celebrating on the soccer field."
        },
        {
            "image": "/path/to/image2.jpeg",
            "question": "Where is this?",
            "basic_answer": "This is at the Allianz Arena in Munich."
        }
    ]
    
    Target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Who are they?",
                    "role": "user"
                },
                {
                    "content": "They're Kane and Gretzka from Bayern Munich.",
                    "role": "assistant"
                },
                {
                    "content": "What are they doing?",
                    "role": "user"
                },
                {
                    "content": "They are celebrating on the soccer field.",
                    "role": "assistant"
                }
            ],
            "images": [
                "/path/to/image1.jpeg"
            ]
        },
        {
            "messages": [
                {
                    "content": "<image>Where is this?",
                    "role": "user"
                },
                {
                    "content": "This is at the Allianz Arena in Munich.",
                    "role": "assistant"
                }
            ],
            "images": [
                "/path/to/image2.jpeg"
            ]
        }
    ]
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                input_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from {input_file}: {e}")
                return
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Group data by image paths
    image_groups = defaultdict(list)
    
    for item in input_data:
        # Skip if question or answer is empty
        if not item.get("question", "").strip() or not item.get("basic_answer", "").strip():
            continue
            
        image_path = item.get("image", "")
        # Group by image path (use "no_image" as key if no image)
        key = image_path if image_path else "no_image"
        image_groups[key].append(item)
    
    output_data = []
    
    # Process each image group as a conversation
    for image_path, items in image_groups.items():
        messages = []
        images = []
        
        # Only add image to the images list if it's a real image path
        if image_path != "no_image":
            images.append(image_path)
        
        first_item = True
        for item in items:
            # For the first item in a group, include the image tag
            if first_item and image_path != "no_image":
                user_content = f"<image>{item['question']}"
                first_item = False
            else:
                # For subsequent items with the same image, don't repeat the image tag
                user_content = item['question']
            
            # Add user message
            messages.append({
                "content": user_content,
                "role": "user"
            })
            
            # Add assistant message
            messages.append({
                "content": item["basic_answer"],
                "role": "assistant"
            })
        
        # Create the new format item
        new_item = {
            "messages": messages,
            "images": images
        }
        
        output_data.append(new_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete:")
    print(f"  Processed {len(input_data)} items")
    print(f"  Created {len(output_data)} conversations")
    print(f"  Output saved to {output_file}")
    # give 1 example
    print(output_data[0])



def run_bash_script(model_name_or_path, 
                    data_path, 
                    output_dir, 
                    template="qwen2_vl",
                    finetuning_type="full",
                    freeze_vision_tower="true",
                    freeze_multi_modal_projector="true",
                    freeze_language_model="false",
                    learning_rate=2e-5,
                    num_train_epochs=3):
    logger.info(f"Finetuning model {model_name_or_path} with template {template}")
    # fix bug;
    # data_tmp = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/planning_maps_data/top1000/caption_new_answer.json"
    # if not os.path.exists(data_tmp):
    #     covert_caption_to_basic_answer(data_path, data_tmp)
    # else:
    #     logger.info(f"Data {data_tmp} already exists")
    convert_format(data_path, "data/selected_data_vlm.json")
    
    bash_command = [
        "bash", "ft_vl_new.sh",  # Changed script name to ft_vl.sh for VL models
        "--model_name_or_path", model_name_or_path,
        "--dataset", "SELECTED_DATA_VLM",
        "--output_dir", output_dir,
        "--template", template,
        "--device", ",".join(gpus),
        "--finetuning_type", finetuning_type,
        "--freeze_vision_tower", freeze_vision_tower,
        "--freeze_multi_modal_projector", freeze_multi_modal_projector,
        "--freeze_language_model", freeze_language_model,
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs)
    ]
    subprocess.run(bash_command)


def sft_vl(data_paths, base_model):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b", "qwen25-vl-7b", "qwen25-vl-3b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2-VL-2B-Instruct",
        "qwen25-vl-7b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2.5-7B-instruct",
        "qwen25-vl-3b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2.5-3B-instruct",
    }
    
    ft_model_path = model_base_path[base_model]
    

    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
        ft_output_dir = f"/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/vlm_train/{base_model}/{name}"
        logger.info(f"Finetuning model {base_model} with data {path}")
    
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template="qwen2_vl"
        )
        


def sft_vl_experiment(data_paths, base_model):
    assert base_model in ["qwen2-vl-7b", "qwen2-vl-2b", "qwen25-vl-7b", "qwen25-vl-3b"], f"Model {base_model} not found in model_base_path"
    
    model_base_path = {
        "qwen2-vl-7b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2-VL-7B-Instruct",
        "qwen2-vl-2b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2-VL-2B-Instruct",
        "qwen25-vl-7b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2.5-7B-instruct",
        "qwen25-vl-3b": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/huggingface_model/Qwen2.5-3B-instruct",
    }
    
    ft_model_path = model_base_path[base_model]
    
    # 实验参数组合
    finetuning_types = ["lora", "full"]
    freeze_combinations = [
        # projector only
        {"vision_tower": "true", "multi_modal_projector": "false", "language_model": "true"},
        # llm only
        {"vision_tower": "true", "multi_modal_projector": "true", "language_model": "false"},
        # projector + llm
        {"vision_tower": "true", "multi_modal_projector": "false", "language_model": "false"}
    ]
    epochs = [2]
    # learning_rates = [ "1e-6", "2e-7","2e-5"]
    learning_rates = ["2e-5"]
    
    
    for path in data_paths:
        name = path.split("/")[-1].split(".")[0]
    
        
        for ft_type in finetuning_types:
            for freeze in freeze_combinations:
                for epoch in epochs:
                    for lr in learning_rates:
                        # 构建输出路径，包含实验参数
                        freeze_str = "v{}_p{}_l{}".format(
                            "f" if freeze["vision_tower"] == "true" else "t",
                            "f" if freeze["multi_modal_projector"] == "true" else "t",
                            "f" if freeze["language_model"] == "true" else "t"
                        )
                        
                        ft_output_dir = f"/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/vlm_train/{base_model}/{name}/{ft_type}_{freeze_str}_ep{epoch}_lr{lr}"
                        logger.info(f"Finetuning model {base_model} with data {path}, type: {ft_type}, freeze: {freeze_str}, epochs: {epoch}, learning_rate: {lr}")
                        
                        run_bash_script(
                            model_name_or_path=ft_model_path,
                            data_path=path,
                            output_dir=ft_output_dir,
                            template="qwen2_vl",
                            finetuning_type=ft_type,
                            freeze_vision_tower=freeze["vision_tower"],
                            freeze_multi_modal_projector=freeze["multi_modal_projector"],
                            freeze_language_model=freeze["language_model"],
                            num_train_epochs=str(epoch),
                            learning_rate=lr
                        )



# qwen2-vl-7b

if __name__ == "__main__":
    
    
    data_path = [
        "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/planvlm_sft_cot_7k/basic_sft_cot_7k.json"
    ]
    
    sft_vl(data_path, "qwen2-vl-7b")
