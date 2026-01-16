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
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Save to {data_path}")
    
    

    
def convert_format(input_file, output_file):
    """
    Convert from the original format to the target format with multiple messages and images,
    grouping conversations by image.
    """
    input_data = load_json(input_file)
    # Group data by image paths
    image_groups = defaultdict(list)
    
    for item in input_data:
        # Skip if question or answer is empty
        if not item.get("instruction", "").strip() or not item.get("response", "").strip():
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
                user_content = f"<image>{item['instruction']}"
                first_item = False
            else:
                # For subsequent items with the same image, don't repeat the image tag
                user_content = item['instruction']
            
            # Add user message
            messages.append({
                "content": user_content,
                "role": "user"
            })
            
            # Add assistant message
            messages.append({
                "content": item["response"],
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
    save_json(output_data, output_file)
    print(output_data[0])
    

