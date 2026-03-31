# 🏙️ PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models

**[📄 Paper (EMNLP 2025 Industry Track)](https://aclanthology.org/2025.emnlp-industry.169/)** |
**[🤖 Model (ModelScope)](https://modelscope.cn/models/chichi56/plangpt-VL-10K)**

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

* Decomposes each question into **verifiable "critical points."**
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
uv sync                    # Install dependencies
uv sync --extra train      # Include training dependencies (LlamaFactory)
```

### 2️⃣ Code Structure

```
PlanGPT-VL/
├── src/
│   ├── core/                       # Configuration and prompt templates
│   │   ├── config.py               # Centralized config (dataclasses)
│   │   └── prompts.py              # All prompt templates (questions, responses, CPT, filtering)
│   ├── tools/                      # Shared infrastructure
│   │   ├── inference_utils.py      # Async inference engine (OpenAI + Azure backends)
│   │   ├── filtering.py            # Planning map detection + resolution filtering
│   │   └── utils.py                # JSON I/O, tag parsing, image utilities
│   ├── pipeline.py                 # PlanAnno-V data synthesis pipeline
│   └── eval/                       # Evaluation system
│       ├── eval.py                 # PlanBench-V core evaluation logic
│       ├── run_planbench_eval.py   # Full evaluation pipeline (vLLM + judge)
│       ├── planbench-subset.json   # PlanBench-V benchmark data (300 questions)
│       └── images/                 # Planning map images for evaluation (93 maps)
├── train/                          # LlamaFactory v0.9.5 (SFT + DPO)
│   ├── configs/                    # Training and merge configs per model
│   └── data/                       # Dataset registration (dataset_info.json)
├── pyproject.toml                  # Project dependencies (uv workspace)
└── docs/                           # Documentation
```

### 3️⃣ Data Synthesis (PlanAnno-V)

```python
from src.pipeline import generate_questions, generate_responses
from src.tools.utils import process_image_directory

# Step 1: Generate questions from planning maps
image_paths = process_image_directory("/path/to/images")
questions = generate_questions(image_paths)

# Step 2: Generate CPT-enhanced responses
responses = generate_responses(questions, mode="direct_cpt")
```

### 4️⃣ Quick Reproduction (LoRA SFT)

This repository provides a **lightweight LoRA reproduction** of PlanGPT-VL using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Pre-built configs are available for multiple Qwen VL model families:

```bash
# Train (edit model path in YAML first)
uv run llamafactory-cli train train/configs/qwen2vl_sft.yaml

# Merge LoRA weights into base model
uv run llamafactory-cli export train/configs/qwen2vl_merge.yaml
```

Available training configs:
| Config | Base Model |
| ------ | ---------- |
| `qwen2vl_sft.yaml` | Qwen2-VL-7B-Instruct |
| `qwen25vl_sft.yaml` | Qwen2.5-VL-7B-Instruct |
| `qwen3vl_sft.yaml` | Qwen3-VL-8B |
| `qwen35_sft.yaml` | Qwen3.5-9B |

> **Note:** The released [PlanGPT-VL model](https://modelscope.cn/models/chichi56/plangpt-VL-10K) was trained with full-parameter SFT on 10K data. The LoRA configs here offer a resource-efficient alternative for quick reproduction on a single GPU.

### 5️⃣ Evaluation

```bash
# PlanBench-V evaluation (requires vLLM for inference + Azure API for judging)
uv run python src/eval/run_planbench_eval.py \
    --model-path /path/to/model \
    --model-name MyModel \
    --model-type qwen2_vl

# Standard benchmarks via lmms-eval
uv run python -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=/path/to/model \
    --tasks ai2d,mmmu_val,mmstar,scienceqa,realworldqa,mathvista_testmini_format \
    --batch_size 1
```

---

## 🔧 Configuration

### Azure OpenAI Endpoints

The inference engine supports Azure OpenAI with multi-endpoint load balancing. Configure via environment variables:

```bash
# Option 1: JSON config file
export AZURE_ENDPOINTS_FILE=/path/to/endpoints.json
export AZURE_TENANT_ID=your-tenant-id

# Option 2: Inline JSON
export AZURE_ENDPOINTS='{"gpt-4o": [{"endpoint": "https://...", "speed": 150, "model": "gpt-4o"}]}'
```

See `docs/azure_endpoints_example.json` for the expected format.

### Model Paths

Training configs use `${MODEL_DIR}` as a placeholder. Set the environment variable or replace with your actual model path before training.

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

## 📜 License

This project is released under the Apache 2.0 License. The LlamaFactory component (`train/`) retains its original Apache 2.0 license.

## 📬 Contact

* **Authors:** He Zhu, Junyou Su, Minxin Chen, Wen Wang, Yijie Deng, Guanhua Chen, Wenjia Zhang
* **Institutions:** Peking University, Tongji University, Southern University of Science and Technology
* **Email:** [zhuye140@gmail.com](mailto:zhuye140@gmail.com) | [wenjiazhang@pku.edu.cn](mailto:wenjiazhang@pku.edu.cn)

---

> **Note:** This repository was first released in October 2025 alongside the EMNLP 2025 publication. The codebase was refactored and reorganized on April 1, 2026.
