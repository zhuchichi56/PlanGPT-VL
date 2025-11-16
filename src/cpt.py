import os
from typing import List, Dict, Any, Tuple
from loguru import logger
import fire
from tqdm import tqdm
from utils import *



PROMPT_TEMPLATE = {
    "critical_version": """作为城市规划专家，请根据已有的图片，问题和回答，提取解决该问题的关键思考点。

问题：{question}

回答：{answer}

请按照以下格式输出：
<thinking>
[1] (描述第一个关键思考点，需具体解释图中的规划要素如何支持这一思考)
[2] (描述第二个关键思考点，需具体解释图中的规划要素如何支持这一思考)
[3] (描述第三个关键思考点，需具体解释图中的规划要素如何支持这一思考)
[4] (如有必要，描述第四个关键思考点)
[5] (如有必要，描述第五个关键思考点)
</thinking>
"""
}



def extract_thinking_content(response):
    thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
    thinking_match = thinking_pattern.search(response)
    if thinking_match:
        return thinking_match.group(1).strip()
    return ""

def combine_thinking_answer(thinking, answer):
    return f"<thinking>\n{thinking}\n</thinking>\n{answer}"


def main(data_path: str = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/tinyvlm/test_vlm_887.json",
         image_base_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/planvlm_eval/data",
         output_path: str = "critical_points_results.json",
         cot_version: str = "critical_version"):

    data = load_json(data_path)
    prompts = []
    image_paths = []
    
    for item in data:
        item["image"] = os.path.join(image_base_dir, item["image_url"])
        image_paths.append(item["image"])
        prompt = PROMPT_TEMPLATE[cot_version].format(
            question=item["question"],
            answer=item["answer"]
        )
        prompts.append(prompt)
        
    results = parallel_image_inference_batch(prompts, image_paths)
    
    for result, item in zip(results, data):
        thinking = extract_thinking_content(result)
        item[f"{cot_version}_response"] = combine_thinking_answer(thinking, item["answer"])

    save_json(results, output_path)
    return results

if __name__ == "__main__":
    main(cot_version="cot_version", output_path="critical_points_results_cot.json")
    main(cot_version="critical_version", output_path="critical_points_results_critical.json")
