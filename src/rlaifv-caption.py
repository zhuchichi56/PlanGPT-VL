import os
import re
from typing import List, Dict, Any, Tuple
from loguru import logger
import fire
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import from the existing files
from utils import * 
from paste import PROMPTS, generate_answers, process_inference
from paste2 import ImageEvaluator, load_json, save_json, API_KEY, API_BASE

def parse_key_points(caption: str) -> List[str]:
    key_points = []
    pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
    matches = re.findall(pattern, caption, re.DOTALL)
    
    for _, content in matches:
        key_point = content.strip()
        if key_point:
            key_points.append(key_point)
    
    if not key_points:
        # For captions without numbering, try to split by newlines
        paragraphs = [p.strip() for p in caption.split('\n\n') if p.strip()]
        if paragraphs:
            key_points = paragraphs
    
    return key_points


def generate_caption(item: Dict[str, Any]) -> str:
    
    temp_item = {
        "image": item["image"],
        "question": f"""请作为城市规划专家，

请提供一个详细的图像描述，遵循以下格式：
[1] 第一个关键点（颜色、区域、分布等）
[2] 第二个关键点（其他规划特征）
[3] 第三个关键点（空间关系）
...

请确保每个关键点都编号并分段，尽可能详细描述图中所见的规划要素，特别是与问题相关的部分。"""
    }
    
    result = generate_answers([temp_item], mode="direct", question_key="question", include_caption=False)[0]
    return result

def improve_key_point(item: Dict[str, Any], key_point: str, index: int) -> str:
    temp_item = {
        "image": item["image"],
        "question": f"""请评估并改进以下关于规划图的描述点，该描述点是为了回答问题："{item["question"]}"

以下是现有的描述点 [#{index}]：
"{key_point}"

请评估此描述点的准确性和完整性，并给出改进建议：
1. 描述是否准确？如果不准确，指出错误之处。
2. 描述是否完整？是否遗漏了重要信息？
3. 描述与问题的相关性如何？是否需要添加更多与问题直接相关的内容？

如果描述已经很好且无需改进，请回复"无需改进"。否则，请提供一个改进的版本，保持相似的长度和风格。

你应该按照如下格式回复：
改进版本：
改进的描述点：
建议修改为：
"""
    }
    
    result = generate_answers([temp_item], mode="direct", question_key="question", include_caption=False)[0]
    
    if "无需改进" in result:
        return ""
    
    improved_point = ""
    patterns = [
        r'改进版本：\s*(.*?)$',
        r'改进的描述点：\s*(.*?)$',
        r'建议修改为：\s*(.*?)$'
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, result, re.DOTALL)
        if matches:
            improved_point = matches.group(1).strip()
            break
    
    if not improved_point:
        lines = result.strip().split('\n')
        for i, line in enumerate(lines):
            if "改进" in line or "建议" in line:
                if i+1 < len(lines):
                    improved_point = lines[i+1].strip()
                    break
    
    return improved_point

def improve_caption(item: Dict[str, Any], current_caption: str) -> str:
    key_points = parse_key_points(current_caption)
    improved_points = []
    
    for i, point in enumerate(key_points):
        improved = improve_key_point(item, point, i+1)
        if improved:
            improved_points.append(f"[{i+1}] {improved}")
        else:
            improved_points.append(f"[{i+1}] {point}")
    
    return "\n\n".join(improved_points)

def iterate_caption(item: Dict[str, Any], max_iterations: int = 3) -> Tuple[str, List[float], List[str]]:
    evaluator = ImageEvaluator(API_KEY, API_BASE)
    current_caption = generate_caption(item)
    scores = []
    all_captions = [current_caption]
    
    # 评估初始caption
    critical_points = item.get("critical_points", ["描述了图中的主要规划区域", "分析了不同功能区的空间分布", "识别了关键的地理要素"])
    score, _ = evaluator.evaluate_answer(
        item["image"], item["question"], critical_points, current_caption, "gpt-4o-mini"
    )
    scores.append(score)
    
    for i in range(max_iterations - 1):
        print(f"Iteration {i+1}/{max_iterations-1} for image {os.path.basename(item['image'])}")
        improved_caption = improve_caption(item, current_caption)
        all_captions.append(improved_caption)
        
        # 评估改进后的caption
        score, _ = evaluator.evaluate_answer(
            item["image"], item["question"], critical_points, improved_caption, "gpt-4o-mini"
        )
        scores.append(score)
        
        # 如果得分没有提高，提前结束
        if len(scores) >= 2 and scores[-1] <= scores[-2]:
            print(f"No improvement in score, stopping at iteration {i+1}")
            break
        
        current_caption = improved_caption
    
    # 返回得分最高的caption
    best_index = scores.index(max(scores))
    return all_captions[best_index], scores, all_captions

def process_dataset(data_path: str, output_path: str, max_iterations: int = 3, max_workers: int = 8):
    data = load_json(data_path)
    results = []
    
    def process_item(item):
        try:
            best_caption, scores, all_captions = iterate_caption(item, max_iterations)
            
            result = item.copy()
            result["caption"] = best_caption
            result["caption_scores"] = scores
            result["all_captions"] = all_captions
            result["best_score"] = max(scores)
            
            print(f"Processed {os.path.basename(item['image'])}, best score: {max(scores):.2f}/2")
            return result
            
        except Exception as e:
            print(f"Error processing item {item.get('id', '')}: {str(e)}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(process_item, item) for item in data]
        
        for future in tqdm(tasks, desc="Processing images"):
            result = future.result()
            if result:
                results.append(result)
    
    save_json(results, output_path)
    print(f"Processed {len(results)} items, saved to {output_path}")
    
    avg_score = sum(item["best_score"] for item in results) / len(results) if results else 0
    print(f"Average best score: {avg_score:.2f}/2")
    
    return results

def main(data_path: str, 
         output_path: str = "caption_results.json",
         max_iterations: int = 3,
         max_workers: int = 8):
    
    return process_dataset(data_path, output_path, max_iterations, max_workers)

if __name__ == "__main__":
    fire.Fire(main)