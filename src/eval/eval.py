
import argparse
import re
import os
import json
import base64
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union
from loguru import logger

DEFAULT_BASE_URL = "http://localhost:8081/v1"
DEFAULT_MODEL = "gpt-4o-mini"

def load_json(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} items to {file_path}")

class ImageEvaluator:
    def __init__(self, api_key: str, api_base: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
    
    @staticmethod
    def encode_image(image_path: str) -> Tuple[str, str]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        image_type = os.path.splitext(image_path)[1].lower()
        mime_type = {".png": "image/png", ".jpg": "image/jpeg", 
                    ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(image_type, "image/jpeg")
        
        return base64.b64encode(content).decode("utf-8"), mime_type
    
    def ask_question(self, model_id: str, image_path: str, question: str,
                    max_retries: int = 3, retry_delay: int = 5) -> str:
        base64_image, mime_type = self.encode_image(image_path)
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "你是一个城市规划评估专家。"}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                {"type": "text", "text": question},
            ]}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=4096,
                )

                return response.choices[0].message.content or ""
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API request failed (attempt {attempt+1}/{max_retries}): {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return f"Error occurred: {str(e)}"
    
    def evaluate_answer(self, image_path: str, question: str, 
                        critical_points: List[str], summary: str,
                        model: str = "gpt-4o-mini") -> Tuple[float, str]:
        """使用关键得分点评估答案"""
        # 将关键得分点合并为文本
        critical_points_text = "\n".join(critical_points)
        
        evaluation_prompt = f"""
请根据问题、得分点列表和图像内容，对下面的回答进行评分。

问题：{question}

得分点列表：
{critical_points_text}

待评估回答：{summary}

评分标准：
请逐一检查模型回答是否涉及到每个得分点：
- 对于每个得分点，如果模型回答中有涉及到相关内容，请给1分
- 如果模型回答中没有涉及到或描述错误，请给0分
- 得分点之间是互斥的，每个得分点最多得1分

请按以下格式进行评分：
1. 得分点1：[0/1] - 简要说明是否包含该得分点及依据
2. 得分点2：[0/1] - 简要说明是否包含该得分点及依据
...

最终得分：X/Y（X为累计得分，Y为总分，即得分点总数）
"""
        # 请求评分
        score_text = self.ask_question(model, image_path, evaluation_prompt)
        
        try:
            # 提取最终得分
            score_match = re.search(r"最终得分：\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                total = float(score_match.group(2))
                
                # 归一化为满分2分
                normalized_score = (score / total) * 2 if total > 0 else 0
                return normalized_score, score_text
            else:
                print(f"Could not extract N_hit/N_total score from response: {score_text}")
                return 0.0, score_text
        except Exception as e:
            print(f"Error parsing N_hit/N_total score: {score_text}, error: {str(e)}")
            return 0.0, score_text


class EvaluationProcessor:
    def __init__(self, api_key: str, api_base: str, model: str = DEFAULT_MODEL):
        self.evaluator = ImageEvaluator(api_key, api_base)
        self.model = model
    
    def process_item(self, item: Dict[str, Any], image_base_dir: str) -> Optional[Dict[str, Any]]:
        try:
            question = item["question"]
            item_type = item.get("type", "")
            image_path = os.path.join(image_base_dir, item["image_url"])
            thinking = item["thinking"]
            summary = item["summary"]
            
            # 获取关键得分点（必须存在）
            critical_points = item.get("critical_points", [])
            if not critical_points:
                print(f"No critical points found for item: {item.get('id', item.get('image_id', 'unknown'))}")
                return None
            
            # 评估答案
            score, score_text = self.evaluator.evaluate_answer(
                image_path,
                question,
                critical_points,
                thinking + "\n" + summary if thinking else summary,
                self.model
            )
            
            item_id = item.get('image_id', item.get('id', 'unknown'))
            print(f"[EVALUATION RESULT] ID: {item_id}, Score: {score}/2")
            
            # 创建结果项
            result = item.copy()
            result["score"] = score
            result["score_text"] = score_text
            return result
            
        except Exception as e:
            print(f"Error processing item {item.get('image_id', item.get('id', 'unknown'))}: {str(e)}")
            return None
    
    def process_dataset(self, json_paths: Union[str, List[str]], 
                        image_base_dir: str = "images", 
                        max_workers: int = 32,
                        save_path: str = "eval_with_critical_points",
                        ) -> Dict[str, List[Dict[str, Any]]]:
        
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        all_results = {}
        
        for json_path in json_paths:

            data = load_json(json_path)
            model_name = os.path.basename(json_path).split('/')[-1].replace(".json", "")
            
            if os.path.exists(os.path.join(save_path, f"{model_name}.json")):
                logger.info(f"Skipping {model_name} because it already exists")
                continue
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [executor.submit(self.process_item, item, image_base_dir) for item in data]
                
                for future in tqdm(tasks, desc=f"Evaluating {model_name}"):
                    result = future.result()
                    if result:
                        results.append(result)
            
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(save_path, f"{model_name}.json")
            
            save_json(results, output_path)
            self.print_summary(results)    
            all_results[model_name] = results
        return all_results
    
    @staticmethod
    def print_summary(results: List[Dict[str, Any]], csv_path: str = "evaluation_summary.csv") -> None:
        import pandas as pd
        import os
        
        # 计算各类型的分数
        type_scores = {}
        for result in results:
            item_type = result.get('type', 'unknown')
            if item_type not in type_scores:
                type_scores[item_type] = []
            type_scores[item_type].append(result['score'])
        
        # 打印摘要
        print("\n===== EVALUATION SUMMARY =====")
        total_scores = []
        summary_data = []
        
        for item_type, scores in type_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"Type '{item_type}' average score: {avg_score:.2f}/2 (total: {len(scores)} items)")
            total_scores.extend(scores)
            summary_data.append({
                'type': item_type,
                'avg_score': avg_score,
                'count': len(scores),
                'model': os.path.basename(results[0].get('file_path', 'unknown')).split('_')[0]
            })
        
        overall_avg = sum(total_scores) / len(total_scores) if total_scores else 0
        print(f"\nOverall average score: {overall_avg:.2f}/2 (total: {len(total_scores)} items)")
        
        # 创建DataFrame
        df = pd.DataFrame(summary_data)
        
        # 修复这里: 检查文件是否存在且非空
        try:
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                try:
                    existing_df = pd.read_csv(csv_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    # 文件存在但是没有数据或格式问题，使用新的DataFrame
                    pass
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {str(e)}")
        
        # 保存DataFrame
        df.to_csv(csv_path, index=False)


def main() -> Dict[str, List[Dict[str, Any]]]:
    parser = argparse.ArgumentParser(description="Evaluate VQA results with proxy.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json-paths", nargs="*", default=None)
    parser.add_argument("--image-base-dir", default="images")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--save-path", default="eval_with_critical_points")
    args = parser.parse_args()

    if args.json_paths is None or len(args.json_paths) == 0:
        default_json = "planbench-subset.json"
        if os.path.exists(default_json):
            json_paths = [default_json]
        else:
            json_paths = [
                "results_planvlm3/question_results_5_13_top1000_answers_cot_merged_wo_type_results_310.json",
                "results_planvlm3/basic_sft_cot_7k_absolute_path_results_310.json",
                "results_planvlm3/Qwen2-VL-7B-Instruct_results_310.json",
                "results_planvlm3/question_results_results_310.json",
            ]
    else:
        json_paths = args.json_paths

    processor = EvaluationProcessor(args.api_key, args.api_base, args.model)
    results = processor.process_dataset(
        json_paths,
        args.image_base_dir,
        args.max_workers,
        save_path=args.save_path,
    )

    print("\nEvaluation completed!")
    return results


if __name__ == "__main__":
    main()
    
    
    
