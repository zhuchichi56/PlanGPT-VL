import json
import os
import shutil
import re
from loguru import logger
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
import threading
import concurrent.futures

# 这个的任务是: 
# 过滤掉分辨率最大的5%的图像，并保留其余的图像 
# 保留所有识别为规划图的图像 

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_planning_maps(input_json, original_data_dir, output_dir, filter_percent=5):
    
    output_images_dir = os.path.join(output_dir, "images")
    output_text_dir = os.path.join(output_dir, "text")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_text_dir, exist_ok=True)

    original_text_dir = os.path.join(original_data_dir, "text")
    text_file_map = {}

    try:
        for filename in os.listdir(original_text_dir):
            if filename.endswith('.txt'):
                hash_code = os.path.splitext(filename)[0]
                text_file_map[hash_code] = os.path.join(original_text_dir, filename)
    except Exception as e:
        logger.error(f"Error accessing text directory: {e}")
        return
    
    data = load_json(input_json)
    planning_maps = [item for item in data if item.get("is_planning_map") == 1]
    logger.info(f"Found {len(planning_maps)} planning maps")
    
    # 获取所有图像的分辨率信息
    image_resolutions = []
    lock = threading.Lock()
    
    def process_image(item):
        try:
            image_path = item["image"]
            img = cv2.imread(image_path)
            if img is not None:
                resolution = img.shape[0] * img.shape[1]  # 高 × 宽 = 总像素数
                with lock:
                    image_resolutions.append((item, resolution))
            else:
                logger.warning(f"Could not read image: {image_path}")
        except Exception as e:
            logger.error(f"Error reading image {item.get('image', 'unknown')}: {e}")
    
    # 使用线程池处理图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        list(tqdm(executor.map(process_image, planning_maps), total=len(planning_maps), desc="Processing images"))
    # 按分辨率降序排序
    image_resolutions.sort(key=lambda x: x[1], reverse=True)
    
    # 计算要过滤掉的图像数量（分辨率最大的5%）
    filter_count = int(len(image_resolutions) * filter_percent / 100)
    logger.info(f"Filtering out {filter_count} images with highest resolution ({filter_percent}%)")
    
    # 过滤掉分辨率最大的5%，只保留其余的图像
    filtered_planning_maps = [item[0] for item in image_resolutions[filter_count:]]
    logger.info(f"Remaining {len(filtered_planning_maps)} planning maps after filtering")
    
    
    # 计算过滤前后的分辨率平均值和最大值
    if image_resolutions:
        original_resolutions = [res for _, res in image_resolutions]
        filtered_resolutions = [res for _, res in image_resolutions[filter_count:]]
        logger.info(f"Original dataset - Average resolution: {sum(original_resolutions)/len(original_resolutions):.2f}, Maximum resolution: {max(original_resolutions)}")
        logger.info(f"Filtered dataset - Average resolution: {sum(filtered_resolutions)/len(filtered_resolutions):.2f}, Maximum resolution: {max(filtered_resolutions)}")
    

    processed_count = 0
    hash_code_set = set()
    
    for item in filtered_planning_maps:
        try:
            image_path = item["image"]
            image_filename = os.path.basename(image_path)
            match = re.match(r'([a-f0-9]+)_figure\d+\.\w+', image_filename)
            if match:
                hash_code = match.group(1)
                hash_code_set.add(hash_code)
            else:
                logger.warning(f"Cannot extract hash code from image filename: {image_filename}")
                continue
            
            dest_image_path = os.path.join(output_images_dir, image_filename)
            shutil.copy2(image_path, dest_image_path)
            
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} planning maps")
                
        except Exception as e:
            logger.error(f"Error processing image {image_path if 'image_path' in locals() else 'unknown'}: {e}")
    
    text_copied_count = 0
    for hash_code in hash_code_set:
        if hash_code in text_file_map:
            try:
                src_text_path = text_file_map[hash_code]
                dest_text_path = os.path.join(output_text_dir, f"{hash_code}.txt")
                shutil.copy2(src_text_path, dest_text_path)
                text_copied_count += 1
            except Exception as e:
                logger.error(f"Error copying text file {hash_code}.txt: {e}")
        else:
            logger.warning(f"No original text file found for hash code {hash_code}")
    
    logger.info(f"Copied {text_copied_count} text files")
    
    summary_path = os.path.join(output_dir, "planning_maps_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_planning_maps, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processing completed. Total processed {processed_count} planning maps")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Summary file: {summary_path}")


def main():
    input_json = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/refined_vllm_results/refined_results.json"
    original_data_dir = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/process_data"
    output_dir = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/planning_maps_data"
    
    filter_planning_maps(input_json, original_data_dir, output_dir, filter_percent=5)


if __name__ == "__main__":
    main()