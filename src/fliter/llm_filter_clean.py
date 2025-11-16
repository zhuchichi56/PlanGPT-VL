import re
import json
import os
from typing import List, Dict, Tuple
from loguru import logger
from inference.client import parallel_image_inference
import fire


# 挑选出这里是1的进行更精细的过滤
# /HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/filter/vllm_results/all_results.json
# # {
#     "image": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/process_data/images/39fa_figure49.jpeg",
#     "analysis": "图像中展示了一张城市道路系统规划图，图中包含了道路网络的布局、图例、比例尺、方向标以及图名等元素。图例中标注了不同类型的道路和交通设施，比例尺显示了地图的比例，方向标指示了地图的方向，图名标注了这是"主城区道路系统规划图"。",
#     "is_planning_map": 1
#   },
#   {
#     "image": "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/process_data/images/cef9_figure79.jpeg",
#     "analysis": "图像中只有一片渐变的浅蓝色背景，没有显示任何空间布局、土地分区、图例、比例尺、方向标、图名或规划单位等元素。",
#     "is_planning_map": 0
#   }
  
  
# 修改后的Prompt，更严格筛选完整的独立规划图
PLANNING_MAP_PROMPT = """你是城市规划专家。请判断下面图像是否为"完整且独立的城市或国土空间规划图"。

请先对图像做一个简要描述，然后判断它是否为规划图。

判断标准：
1. 必须是完整的规划图，而非规划图的一部分或截图
2. 规划图应作为图像的主要内容，占据图像的主要区域
3. 不应包含大量与规划图无关的其他页面元素（如大段文字说明、表格等）
4. 应具备规划图的典型特征：
   - 清晰的空间布局或土地分区的视觉结构
   - 图例、比例尺、方向标、图名、规划单位等规划图必要元素

请注意：
- 如果图像是整页文档的扫描件，且规划图只是页面的一部分，应判定为不符合要求
- 如果图像包含多张规划图，也应判定为不符合要求
- 如果图像内容模糊不清，难以辨认规划内容，应判定为不符合要求

图像分析后，请按以下格式输出:

分析：[在这里提供您的分析]
判断：如果是完整且独立的规划图，请输出：\\boxed{1}
如果不是完整且独立的规划图，请输出：\\boxed{0}
"""

def save_json(data: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        

def parse_result(text: str) -> Tuple[str, int]:
    """Extract both analysis text and boxed value from VLM response."""
    # 提取分析部分
    analysis = ""
    if match := re.search(r'分析：(.*?)(?=判断：|\\boxed{|$)', text, re.DOTALL):
        analysis = match.group(1).strip()
    
    # 提取判断结果
    score = 0
    if match := re.search(r'\\boxed{(\d+)}', text):
        try:
            score = int(match.group(1))
        except ValueError:
            logger.warning(f"Failed to parse score from: {match.group(1)}")
    
    # 如果未找到分析但有其他文本，使用其他文本作为分析
    if not analysis and text:
        # 分割文本，寻找可能的分析内容
        parts = re.split(r'\\boxed{\d+}', text)
        if parts and parts[0].strip():
            analysis = parts[0].strip()
    
    return analysis, score

def evaluate_maps_batch(image_paths: List[str]) -> List[Dict]:
    """Evaluate a batch of images using VLM."""
    results = parallel_image_inference(
        prompts=[PLANNING_MAP_PROMPT] * len(image_paths),
        image_paths=image_paths,
        max_tokens=512,
        temperature=0.1
    )
    
    # 保存原始结果用于调试
    save_data = [{"result": result} for result in results]
    save_json(save_data, "raw_results.json")
    logger.info(f"Received {len(results)} results from inference")
    
    parsed_results = []
    for img, result in zip(image_paths, results):
        if not result:
            logger.warning(f"Empty result for image: {img}")
            parsed_results.append({
                "image": img, 
                "analysis": "", 
                "is_planning_map": 0
            })
            continue
            
        analysis, score = parse_result(result)
        parsed_results.append({
            "image": img,
            "analysis": analysis,
            "is_planning_map": score
        })
    
    return parsed_results

def process_images(input_file: str = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/filter/vllm_results/all_results.json",
                   output_dir: str = "refined_vllm_results", 
                   batch_size: int = 500):
  
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取先前已经过滤的结果
    with open(input_file, 'r', encoding='utf-8') as f:
        previous_results = json.load(f)
    
    # 只保留之前判断为规划图(is_planning_map=1)的图像
    filtered_images = [item["image"] for item in previous_results if item["is_planning_map"] == 1]
    logger.info(f"从先前结果中找到 {len(filtered_images)}/{len(previous_results)} 个初步筛选的规划图")
    
    if not filtered_images:
        logger.warning("没有找到符合条件的图像，请检查输入文件")
        return []
    
    all_results = []
    for i in range(0, len(filtered_images), batch_size):
        batch_images = filtered_images[i:i + batch_size]
        batch_output = os.path.join(output_dir, f"batch_{i//batch_size + 1}_results.json")
        
        if os.path.exists(batch_output):
            logger.info(f"Loading existing batch {i//batch_size + 1} results...")
            with open(batch_output) as f:
                batch_results = json.load(f)
        else:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_images)-1)//batch_size + 1}...")
            batch_results = evaluate_maps_batch(batch_images)
            with open(batch_output, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
                
        all_results.extend(batch_results)
        planning_maps = sum(1 for r in batch_results if r['is_planning_map'])
        logger.info(f"精细筛选后的规划图：{planning_maps}/{len(batch_results)}")
        
        # 输出分析示例
        for idx, result in enumerate(batch_results[:3]):  # 只打印前3个结果作为示例
            logger.info(f"Sample {idx+1}:")
            logger.info(f"  Image: {result['image']}")
            logger.info(f"  Analysis: {result['analysis'][:100]}...")  # 只显示分析的前100个字符
            logger.info(f"  Score: {result['is_planning_map']}")
    
    # Save final results
    final_output = os.path.join(output_dir, "refined_results.json")
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    total_planning = sum(1 for r in all_results if r['is_planning_map'])
    logger.info(f"精细筛选后的总规划图数量: {total_planning}/{len(all_results)}")
    return all_results

if __name__ == "__main__":
    fire.Fire(process_images)