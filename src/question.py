import os
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import fire
import random
import numpy as np
import torch
from utils import * 

PROMPTS = {
    
"context_question_generation_od": """你是一个城市规划师，请观察图片并结合提供的文本上下文提出10个问题：

### 重要:
1. 问题必须基于图片和上下文信息。
2. 从以下15个城市规划维度中任选多个维度，确保问题多样化且专业。
3. 包含7个基础问题（直接从图像可获得答案）和3个扩展延伸问题（需要更深入思考）。
4. 问题长度由短到长，难度由易到难。
5. 必须包含关于空间方位和图例颜色的问题。
6. 把思考过程放在<think></think>中，把问题放在<summary></summary>中。
7. 每个问题前标明是基础问题还是扩展问题，以及从哪个维度提出的。

### 15个城市规划问题维度及示例：

## 1. 图纸类型识别
- 示例："根据图纸的表达风格和内容，判断这是什么类型的规划图？"
- 示例："这张图属于城市总体规划还是详细规划的范畴？"

## 2. 空间关系识别与分析
- 示例："图中主城区与卫星城之间的空间距离如何？"
- 示例："从位置关系看，主要生态区与工业区之间有什么缓冲带？"

## 3. 图例与地理要素识别
- 示例："图中红色区域代表什么功能分区？"
- 示例："图中蓝色线条表示哪些水系要素？"

## 4. 规划方案问题识别与分析
- 示例："规划中工业区紧邻住宅区可能带来哪些问题？"
- 示例："从土地利用效率看，规划方案存在哪些不足？"

## 5. 规划要素描述与定位
- 示例："主要商业中心位于规划区的哪个部分？"
- 示例："图中的水源保护区主要分布在哪些区域？"

## 6. 城市功能区分析
- 示例："不同功能区之间如何实现有效衔接？"
- 示例："城市核心区的功能定位与整体规划目标是否协调？"

## 7. 交通系统规划分析
- 示例："主要交通干道的走向是怎样的？"
- 示例："交通枢纽与各功能区的可达性如何？"

## 8. 生态环境保护规划分析
- 示例："生态廊道在规划中的分布格局是什么？"
- 示例："规划如何保障区域生态系统的完整性？"

## 9. 规划政策与战略关联
- 示例："这份规划反映了哪些国家层面的发展战略？"
- 示例："如何从规划中看出区域协同发展的考量？"

## 10. 历史文化保护与利用
- 示例："历史保护区在城市空间中如何布局？"
- 示例："文化资源保护与城市更新如何平衡？"

## 11. 地图要素识别
- 示例："图中的比例尺和指北针位于何处？"
- 示例："规划图的图例框包含哪些主要元素？"

## 12. 规划图标题推断
- 示例："根据内容，这张图的主题可能是什么？"
- 示例："图中强调的要素表明这是什么性质的规划图？"

## 13. 规划结构分析
- 示例："城市结构是单中心还是多中心模式？"
- 示例："规划中有几条主要发展轴和发展带？"

## 14. 规划影响评估
- 示例："规划实施后可能对区域生态带来什么影响？"
- 示例："这一规划对区域经济格局有何改变作用？"

## 15. 发展协调与平衡分析
- 示例："规划如何处理开发与保护的矛盾？"
- 示例："城市扩张与农田保护如何在规划中取得平衡？"

### 输出格式示例：

<think>
我正在分析一张城市规划图，需要从不同角度提出问题。从图中我观察到以下要素：

基本信息：
- 这是一张覆盖某省域范围的空间规划图
- 图例使用了多种颜色：绿色表示生态保护区域，红色表示城镇建设区，黄色表示农业区
- 有清晰的河流和湖泊系统，呈蓝色
- 有标注的主要城市和重点区域
- 图中存在多条交通线路连接各个中心

空间关系：
- 北部地区以山地为主，多为绿色生态区
- 中部和东部是平原区，以农业和城镇为主
- 南部有大型水体和沿海城市群
- 几个主要城市形成了发展轴带

我将分层次提问：
基础问题(7个)：
- 从图例识别角度：图中颜色含义
- 从地图要素识别角度：指北针和比例尺
- 从空间关系角度：主要城市位置
- 从规划要素描述角度：水系分布
- 从交通系统角度：主要交通廊道
- 从图纸类型识别角度：规划图类型
- 从规划结构分析角度：发展轴和区域数量

扩展问题(3个)：
- 从规划政策角度：与国家战略的关联
- 从生态环境保护角度：生态系统完整性
- 从发展协调与平衡角度：城乡协调发展
</think>

<summary>
【基础问题1】[图例与地理要素识别] 图中绿色、红色和黄色分别代表什么功能区域？

【基础问题2】[地图要素识别] 规划图的指北针位于哪个位置？

【基础问题3】[空间关系识别] 主要城市群分布在规划区的哪个方位？

【基础问题4】[规划要素描述] 图中的主要水系如何分布？

【基础问题5】[交通系统规划] 规划中的主要交通廊道连接了哪些重要节点？

【基础问题6】[图纸类型识别] 根据表达内容和范围，这属于哪种类型的规划图？

【基础问题7】[规划结构分析] 规划区内有几个主要发展轴和功能分区？它们的空间布局特点是什么？

【扩展问题8】[规划政策与战略关联] 从规划的空间布局来看，这份规划如何回应国家关于区域协调发展的战略要求？

【扩展问题9】[生态环境保护规划分析] 规划中的生态保护区域构成了怎样的网络系统？这种布局对维护区域生态安全具有什么作用？

【扩展问题10】[发展协调与平衡分析] 规划中如何处理城镇开发边界扩张与永久基本农田保护之间的潜在冲突？城乡融合发展的思路体现在哪些空间安排上？
</summary>
"""
}
def parse_questions(text: str) -> List[Dict[str, str]]:
    """从文本中解析问题列表，返回包含问题类型、维度和内容的字典列表"""
    # 匹配新格式的问题：【基础问题1】[图例与地理要素识别] 问题内容
    question_matches = re.findall(r'【([^】]+)】\s*\[([^\]]+)\]\s*(.*?)$', text, re.MULTILINE)
    
    parsed_questions = []
    for match in question_matches:
        if len(match) == 3:
            question_type, dimension, content = match
            parsed_questions.append({
                "type": question_type,
                "dimension": dimension,
                "content": content.strip()
            })
    
    return parsed_questions

def generate_qa(image_paths: List[str]) -> List[Dict]:
    """基于图片和描述生成问题"""
    results = process_inference("context_question_generation_od", image_paths, PROMPTS=PROMPTS)
    all_questions = []
    
    for image_path, result in zip(image_paths, results):
        parsed_questions = parse_questions(result)
        
        for question in parsed_questions:
            question_data = {
                "image": image_path,
                "question": question["content"],
                "type": question["type"],
                "dimension": question["dimension"]
            }
            all_questions.append(question_data)
            
    return all_questions


# 甚至都不用过滤了; 
def main(image_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/results/planning_maps_data/top1000/images",
         output_path: str = "/HOME/sustc_ghchen/sustc_ghchen_4/planvlmcore/results"):
    """主函数，按照正确的流程处理数据"""
    image_paths = process_image_directory(image_dir)
    import random
    random.shuffle(image_paths)
    question_results = generate_qa(image_paths)
    save_json(question_results, os.path.join(output_path, "question_results_5_13_top1000.json"))
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
    

