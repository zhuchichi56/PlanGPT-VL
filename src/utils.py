import json

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import fire
from inference.client import parallel_image_inference
import random
import numpy as np
import torch
from inference.client import parallel_image_inference

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def parse_sections(text: str) -> Dict[str, str]:
    """从响应中解析思考和总结部分"""
    result = {"thinking": "", "summary": ""}
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match: result["thinking"] = think_match.group(1).strip()
    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if summary_match: result["summary"] = summary_match.group(1).strip()
    return result

def parse_questions(text: str) -> List[str]:
    """从文本中解析问题列表"""
    question_matches = re.findall(r'问题\d+[\.:]?\s*(.*?)$|^\d+\.?\s*(.*?)$', text, re.MULTILINE)
    return [match[0].strip() if match[0] else match[1].strip() for match in question_matches]

def process_image_directory(directory_path: str) -> List[str]:
    """处理图片目录，返回所有图片的路径"""
    if not os.path.exists(directory_path): return []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


from loguru import logger
def parallel_image_inference_batch(prompt_list: List[str], 
                      image_paths: List[str], 
                      batch_size: int = 200, 
                      output_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/tmp2",
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      top_k: int = 50,
                      max_tokens: int = 4096) -> List[Dict]:
    

    assert len(prompt_list) == len(image_paths), f"prompt_list长度({len(prompt_list)})与image_paths长度({len(image_paths)})不匹配"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_file = os.path.join(output_dir, f"inference_checkpoint.json")
    else:
        checkpoint_file = None
    
    completed_indices = set()
    all_results = [None] * len(image_paths)
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_json(checkpoint_file)
            completed_indices = set(checkpoint_data["completed_indices"])
            
            for idx, result in checkpoint_data["results"].items():
                idx = int(idx)
                if idx < len(all_results):
                    all_results[idx] = result
                    
            logger.info(f"已从断点恢复 {len(completed_indices)} 个结果")
        except Exception as e:
            logger.error(f"恢复断点失败: {e}")
    
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        
        batch_indices = list(range(start_idx, end_idx))
        pending_indices = [idx for idx in batch_indices if idx not in completed_indices]
        
        if not pending_indices:
            logger.info(f"批次 {batch_idx+1}/{total_batches} 已处理过，跳过")
            continue
        
        logger.info(f"处理批次 {batch_idx+1}/{total_batches}，共 {len(pending_indices)} 个项目")
        

        batch_image_paths = [image_paths[idx] for idx in pending_indices]
        batch_prompts = [prompt_list[idx] for idx in pending_indices]
        
        try:
            batch_results = parallel_image_inference(
                batch_prompts,
                batch_image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            for i, idx in enumerate(pending_indices):
                all_results[idx] = batch_results[i]
                completed_indices.add(idx)
            
            if checkpoint_file:
                results_dict = {str(i): res for i, res in enumerate(all_results) if res is not None}
                checkpoint_data = {
                    "completed_indices": list(completed_indices),
                    "results": results_dict
                }
                save_json(checkpoint_data, checkpoint_file)
                logger.info(f"已保存断点，当前已完成 {len(completed_indices)}/{len(image_paths)}")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx+1} 处理失败: {e}")
    
    # 检查是否有未完成的项目
    if len(completed_indices) < len(image_paths):
        logger.warning(f"有 {len(image_paths) - len(completed_indices)} 个项目未完成处理")
    
    return all_results





def process_inference(prompt_template: str, 
                      image_paths: List[str], 
                      params: List[Dict] = None, 
                      batch_size: int = 200, 
                      output_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/tmp2",
                      PROMPTS: Dict[str, str] = None) -> List[Dict]:
    
    if not image_paths: 
        return []
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_file = os.path.join(output_dir, f"{prompt_template}_checkpoint.json")
    else:
        checkpoint_file = None
    
    template = PROMPTS.get(prompt_template, prompt_template)
    prompts = [template.format(**p) for p in params] if params else [template] * len(image_paths)
    
    # 检查是否有恢复点
    completed_indices = set()
    all_results = [None] * len(image_paths)
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_json(checkpoint_file)
            completed_indices = set(checkpoint_data["completed_indices"])
            
            # 恢复已完成的结果
            for idx, result in checkpoint_data["results"].items():
                idx = int(idx)
                if idx < len(all_results):
                    all_results[idx] = result
                    
            logger.info(f"已从断点恢复 {len(completed_indices)} 个结果")
        except Exception as e:
            logger.error(f"恢复断点失败: {e}")
    
    # 分批处理
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        
        # 跳过已完成的批次
        batch_indices = list(range(start_idx, end_idx))
        pending_indices = [idx for idx in batch_indices if idx not in completed_indices]
        
        if not pending_indices:
            logger.info(f"批次 {batch_idx+1}/{total_batches} 已处理过，跳过")
            continue
        
        logger.info(f"处理批次 {batch_idx+1}/{total_batches}，共 {len(pending_indices)} 个项目")
        
        # 准备当前批次的数据
        batch_image_paths = [image_paths[idx] for idx in pending_indices]
        batch_prompts = [prompts[idx] for idx in pending_indices]
        
        try:
            # 执行推理
            batch_results = parallel_image_inference(
                batch_prompts,
                batch_image_paths,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9
            )
            
            # 更新结果和完成索引
            for i, idx in enumerate(pending_indices):
                all_results[idx] = batch_results[i]
                completed_indices.add(idx)
            
            # 保存断点
            if checkpoint_file:
                results_dict = {str(i): res for i, res in enumerate(all_results) if res is not None}
                checkpoint_data = {
                    "completed_indices": list(completed_indices),
                    "results": results_dict
                }
                save_json(checkpoint_data, checkpoint_file)
                logger.info(f"已保存断点，当前已完成 {len(completed_indices)}/{len(image_paths)}")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx+1} 处理失败: {e}")
            # 如果出错，继续下一批次，不中断整个流程
    
    # 检查是否有未完成的项目
    if len(completed_indices) < len(image_paths):
        logger.warning(f"有 {len(image_paths) - len(completed_indices)} 个项目未完成处理")
    
    return all_results




def load_jsonlines(file_path):
    """
    加载jsonlines文件，转换为json格式
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path):
    """
    加载json文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """
    保存为json文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_jsonlines(data, file_path):
    """
    保存为jsonlines文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                