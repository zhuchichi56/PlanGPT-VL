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
