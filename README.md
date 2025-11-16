# 🏙️ PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models

**[📄 Paper (EMNLP 2025 Industry Track)](https://aclanthology.org/2025.emnlp-industry.169/)**
**[🤗 Hugging Face Model Page](https://huggingface.co/chichi56/plangpt-VL-10K)**

---

## 🌆 Overview

**PlanGPT-VL** is the **first domain-specific Vision-Language Model (VLM)** designed for **urban planning map interpretation**.
It bridges the gap between general multimodal AI and professional spatial analysis, enabling planners, policymakers, and educators to **understand, evaluate, and reason over urban planning maps** with expert-level precision.

> 🧭 *Published at EMNLP 2025 (Industry Track), this work introduces a scalable framework for domain-specific multimodal intelligence in spatial planning.*

---

## 🚀 Key Innovations

### 1️⃣ PlanAnno-V Framework

A structured **data synthesis pipeline** that generates high-quality instruction–response pairs from real planning maps.

* Collects and filters over **5,000 urban planning maps**.
* Combines **expert annotations**, **automated instruction expansion**, and **professional tone alignment**.
* Produces a **10K+ multimodal dataset** for instruction tuning.

### 2️⃣ Critical Point Thinking (CPT)

A novel **Generate–Verify–Revise** paradigm to reduce hallucination in complex visual reasoning.

* Decomposes each question into **verifiable “critical points.”**
* Performs iterative **verification and correction**.
* Reduces hallucination by up to **19.2%** in implementation tasks.

### 3️⃣ PlanBench-V Benchmark

The **first benchmark** designed for evaluating VLMs in **urban planning contexts**.

* 300+ curated examples from **real planning maps**.
* Evaluates **Perception**, **Reasoning**, **Association**, and **Implementation** capabilities.
* Scored via multi-dimensional expert criteria for professional validity.

---

## 🧠 Model Architecture & Training

| Component         | Description                                               |
| ----------------- | --------------------------------------------------------- |
| **Base Model**    | Qwen2-VL-7B-Instruct                                      |
| **Framework**     | [VERL](https://github.com/OpenRLHF/VERL)                  |
| **Training Data** | 10K instruction–response pairs from PlanAnno-V            |
| **Training GPUs** | 4× NVIDIA A100                                            |
| **Loss**          | Supervised Fine-Tuning (SFT) with rejection sampling      |
| **Frozen Layers** | Vision encoder & projector (to retain general capability) |

---

## 📊 Experimental Results

| **Model**                           | **Perc**  | **Reas**  | **Assoc** | **Impl**  | **Overall**        |
| ----------------------------------- | --------- | --------- | --------- | --------- | ------------------ |
| Qwen2-VL-2B-Instruct                | 0.767     | 0.664     | 0.926     | 0.616     | 0.731 (-0.179)     |
| Qwen2-VL-7B-Instruct *(base)*       | 0.964     | 0.878     | 0.979     | 0.795     | 0.910              |
| Qwen2.5-VL-7B-Instruct              | 1.168     | 1.013     | 1.069     | 0.880     | 1.050 (+0.140)     |
| Qwen2.5-VL-32B-Instruct *(teacher)* | 1.496     | 1.649     | 1.685     | 1.660     | 1.616              |
| GPT-4o                              | 1.136     | 1.399     | 1.527     | 1.287     | 1.342 (+0.432)     |
| **PlanGPT-VL-2B**                   | 1.247     | 1.366     | 1.453     | 1.386     | 1.352 (+0.442)     |
| **PlanGPT-VL-7B**                   | **1.492** | **1.627** | **1.537** | **1.520** | **1.566 (+0.656)** |


> PlanGPT-VL outperforms open-source and commercial models by **59.2%** on average, with a **7B model** achieving results comparable to **32B-parameter** systems.

---

## 🧩 PlanBench-V Categories

| Main Category      | Subtasks                                                  | Description                              |
| ------------------ | --------------------------------------------------------- | ---------------------------------------- |
| **Perception**     | Element Recognition, Description                          | Identify and describe spatial elements   |
| **Reasoning**      | Classification, Spatial Relations, Professional Reasoning | Perform expert-level spatial analysis    |
| **Association**    | Policy Integration                                        | Link map content to planning regulations |
| **Implementation** | Evaluation, Decision-Making                               | Assess and recommend planning strategies |

---

## ⚙️ Quick Start

### 1️⃣ Installation

```bash
git clone https://github.com/zhuchichi56/PlanGPT-VL.git
cd PlanGPT-VL
pip install -r requirements.txt
```

### 2️⃣ Code Structure

The codebase is organized into modular components for easy extension and maintenance:

```
src/
├── inference/          # VLM Inference Server (vLLM-based)
│   ├── server.py       # FastAPI inference server
│   ├── client.py       # Client with load balancing
│   └── start.py        # Multi-GPU server management
│
├── core/               # Core Configuration
│   ├── prompts.py      # All prompt templates (preserved exactly)
│   └── config.py       # Configuration management
│
├── common/             # Shared Utilities
│   ├── io_utils.py     # JSON/JSONLINES I/O
│   ├── image_utils.py  # Image processing
│   ├── text_utils.py   # Text parsing
│   └── inference_utils.py  # Batch inference with checkpoints
│
├── data_processing/    # Data Generation
│   ├── question_generator.py   # Question generation
│   ├── response_generator.py   # Response generation
│   └── cpt_generator.py        # Critical Point Thinking
│
├── filtering/          # Image Filtering
│   ├── planning_map_filter.py  # Planning map detection
│   └── resolution_filter.py    # Resolution-based filtering
│
├── analysis/           # Analysis & Post-processing
│   ├── postprocessor.py    # Dataset statistics & visualization
│   └── caption_refiner.py  # RLAIF-V caption refinement
│
└── scripts/            # Entry Point Scripts
    ├── generate_questions.py
    ├── generate_responses.py
    └── filter_images.py
```

### 3️⃣ Usage Examples

#### Start Inference Server

```bash
# Single-GPU server
cd src/inference
python start.py \
  --model_path /path/to/Qwen2.5-VL-32B-Instruct \
  --gpu_ids "0" \
  --port 8000

# Multi-GPU server (4 GPUs, tensor parallelism)
python start.py \
  --model_path /path/to/Qwen2.5-VL-32B-Instruct \
  --tensor_parallel_size 4 \
  --gpu_ids "0,1,2,3" \
  --port 8000
```

#### Generate Questions

```bash
cd src
python -m scripts.generate_questions \
  --image_dir /path/to/planning_maps \
  --output questions.json \
  --batch_size 200
```

#### Generate Responses

```bash
cd src
python -m scripts.generate_responses \
  --input questions.json \
  --output responses.json \
  --mode direct_cpt \
  --batch_size 200
```

#### Filter Planning Maps

```bash
cd src
python -m scripts.filter_images \
  --input_dir /path/to/images \
  --output filtered_results.json \
  --batch_size 500
```

#### Programmatic Usage

```python
# Question Generation
from data_processing import generate_questions
from common import process_image_directory

image_paths = process_image_directory("/path/to/images")
questions = generate_questions(image_paths, batch_size=200)

# Response Generation
from data_processing import generate_responses
responses = generate_responses(questions, mode="direct_cpt")

# Filtering
from filtering import filter_planning_maps
results = filter_planning_maps(image_paths)
```

---

## 📈 Dataset Summary

| Aspect               | Description                                 |
| -------------------- | ------------------------------------------- |
| **Total Maps**       | 5,000 collected from urban planning bureaus |
| **Filtered**         | 1,050 representative, 800 expert-annotated  |
| **Synthesized Data** | 10K instruction–response pairs              |
| **Quality Metrics**  | Cosine Similarity = 0.935, MMD = 0.0515     |
| **Language**         | Chinese & English bilingual planning data   |

---

## ⚖️ Limitations & Future Work

* Trade-off between **domain specialization** and **general multimodal performance**.
* Current dataset primarily represents **Chinese urban planning**.
* Future goals include:

  * Expanding to **international datasets**.
  * Incorporating **reinforcement learning** for factual consistency.
  * Developing **cross-domain VLMs** for architecture, environment, and policy analysis.

---


## 🧾 Citation

If you use PlanGPT-VL in your research, please cite:

```bibtex
@inproceedings{zhu2025plangptvl,
  title={PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models},
  author={He Zhu and Junyou Su and Minxin Chen and Wen Wang and Yijie Deng and Guanhua Chen and Wenjia Zhang},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year={2025},
  pages={2461--2483},
  url={https://aclanthology.org/2025.emnlp-industry.169/}
}
```

---

## 📬 Contact

* **Authors:** He Zhu, Junyou Su, Minxin Chen, Wen Wang, Yijie Deng, Guanhua Chen, Wenjia Zhang
* **Institutions:** Peking University, Tongji University, Southern University of Science and Technology
* **Email:** [zhuye140@gmail.com](mailto:zhuye140@gmail.com) | [wenjiazhang@pku.edu.cn](mailto:wenjiazhang@pku.edu.cn)
