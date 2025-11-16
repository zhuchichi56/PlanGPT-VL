import subprocess
from loguru import logger
import os
import json
import json
import os
from collections import defaultdict
import random
random.seed(42)

gpus = ["0", "1","2","3"]
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
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Save to {data_path}")
    
    
def check_images(input_file):
    from tqdm import tqdm
    import os
    
    input_data = load_json(input_file)
    images_check = []
    for item in input_data:
        image_path = item.get("image", "") or item.get("images", [])[0]
        if image_path:
            images_check.append(image_path)
    print(f"Total images to check: {len(images_check)}")
    
    to_check = list(set(images_check))
    print(f"Unique images to check: {len(to_check)}")
    
    images_not_found = []
    for image_path in tqdm(to_check, desc="Checking images"):
        # 使用os.path.exists进行快速检查，比打开图片更高效
        if not os.path.exists(image_path):
            images_not_found.append(image_path)
            print(f"Image not found: {image_path}")
            continue
            
        # 只对存在的文件尝试打开，验证是否为有效图片
        try:
            from PIL import Image
            Image.open(image_path).verify()  # 使用verify()而不是完全加载图片
        except Exception as e:
            print(f"Error validating image {image_path}: {e}")
            images_not_found.append(image_path)
    
    print(f"Images not found or invalid: {len(images_not_found)}")
    return images_not_found
            


            
            
def convert_format(input_file, output_file, question_key="question", response_key="response"):
    """
    Convert from the original format to the target format with multiple messages and images,
    grouping conversations by image.
    """
    input_data = load_json(input_file)
    # check_images(input_file)
    
    
    # Group data by image paths
    image_groups = defaultdict(list)
    
    for item in input_data:
        # Skip if question or answer is empty
        if not item.get(question_key, "").strip() or not item.get(response_key, "").strip():
            continue
            
        image_path = item.get("image", "")
        # Group by image path (use "no_image" as key if no image)
        key = image_path if image_path else "no_image"
        image_groups[key].append(item)
        
    # 对每一轮对话的顺序都随机
    for key in image_groups:
        random.shuffle(image_groups[key])
        
    
    
    
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
            if first_item and image_path != "no_image":
                user_content = f"<image>{item[question_key]}"
                first_item = False
            else:
                # For subsequent items with the same image, don't repeat the image tag
                user_content = item[question_key]
            
            # Add user message
            messages.append({
                "content": user_content,
                "role": "user"
            })
            
            # Add assistant message
            messages.append({
                "content": item[response_key],
                "role": "assistant"
            })
        
        # Create the new format item
        new_item = {
            "messages": messages,
            "images": images
        }
        
        output_data.append(new_item)
        
    # output_data = random.sample(output_data,500)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    save_json(output_data, output_file)
    print(f"Conversion complete:")
    print(f"  Processed {len(input_data)} items")
    print(f"  Created {len(output_data)} conversations")
    print(f"  Output saved to {output_file}")
    print(output_data[0])
    

# def convert_format_single(input_file, output_file):
#     """
#     Convert from the original format to the target format with multiple messages and images,
#     grouping conversations by image.
#     """
#     input_data = load_json(input_file)
#     output_data = []
    
#     for item in input_data:
        
#         messages = []
#         images = []    
        
#         user_content = f"<image>请你仔细描述这张规划图"
#         response = item["caption"]["caption"]
        
#         messages.append({
#             "content": user_content,
#             "role": "user"
#         })
#         messages.append({
#             "content": response,
#             "role": "assistant"
#         })
#         # Only add image to the images list if it's a real image path
#         image_path = item.get("image", "")
#         if image_path:
#             images.append(image_path)
#         # Create the new format item
#         new_item = {
#             "messages": messages,
#             "images": images
#         }
#         output_data.append(new_item)
    
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
#     save_json(output_data, output_file)
#     print(f"Conversion complete:")
#     print(f"  Processed {len(input_data)} items")
#     print(f"  Created {len(output_data)} conversations")
#     print(f"  Output saved to {output_file}")
#     print(output_data[0])

def combine_data(data_path1 , data_path2, output_file):
    data1 = load_json(data_path1)
    data2 = load_json(data_path2)
    print("data1", len(data1))
    print("data2", len(data2))
    output_data = data1 + data2
    import random
    random.shuffle(output_data)
    print(f"Length of combined data: {len(output_data)}")
    save_json(output_data, output_file)

# show
# def run_bash_script(model_name_or_path, 
#                     data_path, 
#                     output_dir, 
#                     template="qwen2_vl",
#                     finetuning_type="full",
#                     freeze_vision_tower="true",
#                     freeze_multi_modal_projector="false",
#                     freeze_language_model="false",
#                     learning_rate=2e-5,
#                     num_train_epochs=3):
#     logger.info(f"Finetuning model {model_name_or_path} with template {template}")
#     logger.info(f"Convert format from {data_path} to {output_dir}")
#     logger.info(f"Length of data: {len(load_json(data_path))}")
#     # convert_format(data_path, "data/selected_data_vlm.json")
    
#     convert_format(data_path, "data/selected_data_vlm1.json")
#     domain_path = "data/selected_data_vlm1.json"
#     general_path = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/ShareGPT4V/coco_items_5k_train2017_converted.json"
#     combine_data(domain_path, general_path, "data/selected_data_vlm.json")
#     check_images("data/selected_data_vlm.json")
    
#     bash_command = [
#         "bash", "ft_vl_new.sh",  # Changed script name to ft_vl.sh for VL models
#         "--model_name_or_path", model_name_or_path,
#         "--dataset", "SELECTED_DATA_VLM",
#         "--output_dir", output_dir,
#         "--template", template,
#         "--device", ",".join(gpus),
#         "--finetuning_type", finetuning_type,
#         "--freeze_vision_tower", freeze_vision_tower,
#         "--freeze_multi_modal_projector", freeze_multi_modal_projector,
#         "--freeze_language_model", freeze_language_model,
#         "--learning_rate", str(learning_rate),
#         "--num_train_epochs", str(num_train_epochs)
#     ]
#     subprocess.run(bash_command)

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
    logger.info(f"Convert format from {data_path} to {output_dir}")
    logger.info(f"Length of data: {len(load_json(data_path))}")
    
    # convert_format_single(data_path, "data/selected_data_vlm.jsonl")
    
    convert_format(data_path, "data/selected_data_vlm.json")
    # data = load_json(data_path)
    # save_json(data, "data/selected_data_vlm.jsonl")
    
    
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
    epochs = [1]
    learning_rates = ["2e-5"]
     # learning_rates = [ "1e-6", "2e-7","2e-5"]
    
    
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





def sft_vl(data_paths, base_model, freeze_projector="true"):
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
        ft_output_dir = f"/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/vlm_train/{base_model}/{name}_freeze_projector_{freeze_projector}_500"
        logger.info(f"Finetuning model {base_model} with data {path}")
    
        run_bash_script(
            model_name_or_path=ft_model_path,
            data_path=path,
            output_dir=ft_output_dir,
            template="qwen2_vl",
            freeze_multi_modal_projector=freeze_projector
        )
        

if __name__ == "__main__":    
    data_path = [
        # "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/results/question_results_5_13_top1000_answers_cot_merged_wo_type.json"
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/image_caption_caption_results_1000.json"
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000.json"
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000.json",
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/question_results_5_13_top1000_answers_cpt_v2.json" # 第一次用cpt
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000_1.json",
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000_qwen32.json",
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000_qwen32_long.json"
        # "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000_qwen32_long.json"
        "/HOME/sustc_ghchen/sustc_ghchen_4/PlanVLM-SFT-DATA/final_results_1000_qwen32_long_wo_caption.json"
    ]
    
    
    sft_vl(data_path, "qwen2-vl-7b", freeze_projector="true")
    
    

