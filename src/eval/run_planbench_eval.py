#!/usr/bin/env python3
"""
PlanBench-V evaluation for local VLM models.

Two-step pipeline:
1. Generate answers (thinking + summary) using local VLM via vLLM
2. Judge/score answers using OpenAI-compatible API (e.g., Azure OpenAI)

Usage:
    # Evaluate a single model
    python run_planbench_eval.py \
        --model-path /path/to/model \
        --model-name Qwen2-VL-7B-base \
        --model-type qwen2_vl

    # Evaluate all models sequentially
    python run_planbench_eval.py --run-all

Environment variables:
    AZURE_TENANT_ID: Azure AD tenant ID (required for Azure judge)
    AZURE_ENDPOINTS_FILE: Path to JSON file with Azure endpoint configs
    VLLM_PYTHON: Path to vLLM Python binary (default: python3)
    MODEL_DIR: Base directory for model weights (default: ./models)
"""

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "planbench-subset.json")
DEFAULT_IMAGE_BASE = SCRIPT_DIR

# Judge API configuration (loaded from environment)
TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")


def _load_judge_endpoints() -> list:
    """Load judge API endpoints from environment config."""
    config_path = os.environ.get("AZURE_ENDPOINTS_FILE")
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
        return data.get("gpt-4o-mini", [])

    inline = os.environ.get("AZURE_ENDPOINTS")
    if inline:
        data = json.loads(inline)
        return data.get("gpt-4o-mini", [])

    # Fallback: use JUDGE_API_BASE if set
    base_url = os.environ.get("JUDGE_API_BASE")
    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    if base_url:
        return [{"endpoint": base_url, "model": model}]

    return []


GPT4O_MINI_ENDPOINTS = _load_judge_endpoints()

# Model configurations for --run-all
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.expanduser("~"), "models"))


def _build_all_models() -> list:
    """Build model configurations, using MODEL_DIR for base model paths."""
    return [
        {
            "model_path": os.path.join(MODEL_DIR, "Qwen2-VL-7B-Instruct"),
            "model_name": "Qwen2-VL-7B-base",
            "model_type": "qwen2_vl",
        },
        {
            "model_path": os.path.join(PROJECT_DIR, "outputs/qwen2vl7b-sft/merged"),
            "model_name": "Qwen2-VL-7B-SFT",
            "model_type": "qwen2_vl",
        },
        {
            "model_path": os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct"),
            "model_name": "Qwen2.5-VL-7B-base",
            "model_type": "qwen2_5_vl",
        },
        {
            "model_path": os.path.join(PROJECT_DIR, "outputs/qwen25vl7b-sft/merged"),
            "model_name": "Qwen2.5-VL-7B-SFT",
            "model_type": "qwen2_5_vl",
        },
        {
            "model_path": os.path.join(MODEL_DIR, "Qwen3-VL-8B-Instruct"),
            "model_name": "Qwen3-VL-8B-base",
            "model_type": "qwen3_vl",
        },
        {
            "model_path": os.path.join(PROJECT_DIR, "outputs/qwen3vl8b-sft/merged"),
            "model_name": "Qwen3-VL-8B-SFT",
            "model_type": "qwen3_vl",
        },
        {
            "model_path": os.path.join(MODEL_DIR, "Qwen3.5-9B"),
            "model_name": "Qwen3.5-9B-base",
            "model_type": "qwen3_5",
        },
        {
            "model_path": os.path.join(PROJECT_DIR, "outputs/qwen35-9b-sft/merged"),
            "model_name": "Qwen3.5-9B-SFT",
            "model_type": "qwen3_5",
        },
    ]


ALL_MODELS = _build_all_models()

VLLM_PORT = 8100  # Use non-default port to avoid conflicts
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", "python3")


def encode_image(image_path: str) -> Tuple[str, str]:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        content = f.read()
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/jpeg")
    return base64.b64encode(content).decode("utf-8"), mime_type


# ============================================================
# vLLM Server Management
# ============================================================

def start_vllm_server(model_path: str, port: int = VLLM_PORT,
                      gpu_mem_util: float = 0.90,
                      max_model_len: int = 16384) -> subprocess.Popen:
    """Start a vLLM server for the given model."""
    cmd = [
        VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", "eval-model",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--limit-mm-per-prompt", '{"image": 5}',
        "--dtype", "bfloat16",
    ]

    log_path = f"/tmp/vllm_planbench_{port}.log"
    log_file = open(log_path, "w")

    print(f"  Starting vLLM server: {model_path}")
    print(f"  Log: {log_path}")

    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    import urllib.request
    health_url = f"http://localhost:{port}/v1/models"
    max_wait = 600  # 10 minutes (some models need extra compile time)
    start = time.time()

    while time.time() - start < max_wait:
        try:
            req = urllib.request.urlopen(health_url, timeout=5)
            if req.status == 200:
                print(f"  vLLM server ready! (took {time.time()-start:.0f}s)")
                return proc
        except Exception:
            pass

        # Check if process died
        if proc.poll() is not None:
            log_file.close()
            with open(log_path) as f:
                tail = f.read()[-2000:]
            raise RuntimeError(f"vLLM server died. Last log:\n{tail}")

        time.sleep(5)

    # Timeout — kill and raise
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    raise RuntimeError(f"vLLM server failed to start within {max_wait}s")


def stop_vllm_server(proc: subprocess.Popen):
    """Stop the vLLM server."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("  vLLM server stopped.")


# ============================================================
# Step 1: Generate answers using local VLM
# ============================================================

def generate_answer_vllm(client, image_path: str, question: str,
                         max_retries: int = 3) -> Tuple[str, str]:
    """Generate answer using local vLLM model."""
    b64_image, mime_type = encode_image(image_path)

    prompt = f"""请你仔细观察图片，然后回答以下问题。

问题：{question}

请按照以下格式回答：

<think>
（在这里写出你的详细分析过程）
</think>

<summary>
（在这里写出你的最终答案摘要）
</summary>
"""

    messages = [
        {"role": "system", "content": "你是一个城市规划评估专家。请仔细分析图片内容，给出专业的回答。"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
            {"type": "text", "text": prompt},
        ]}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="eval-model",
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or ""

            # Parse thinking and summary
            think_match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", raw, re.DOTALL)
            summary_match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)

            thinking = think_match.group(1).strip() if think_match else ""
            summary = summary_match.group(1).strip() if summary_match else ""

            # Fallback: if no tags, use full response as summary
            if not summary:
                summary = raw.strip()

            return thinking, summary

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [Retry {attempt+1}] {e}")
                time.sleep(3)
            else:
                return "", f"[ERROR] {e}"


def generate_all_answers(dataset: List[Dict], image_base: str,
                         port: int = VLLM_PORT) -> List[Dict]:
    """Generate answers for all items using local vLLM."""
    from openai import OpenAI

    client = OpenAI(
        api_key="dummy",
        base_url=f"http://localhost:{port}/v1",
    )

    results = []
    for i, item in enumerate(tqdm(dataset, desc="Generating answers")):
        image_path = os.path.join(image_base, item["image_url"])
        if not os.path.exists(image_path):
            print(f"  [SKIP] Image not found: {image_path}")
            continue

        thinking, summary = generate_answer_vllm(client, image_path, item["question"])

        result = item.copy()
        result["thinking"] = thinking
        result["summary"] = summary
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{len(dataset)} answers")

    return results


# ============================================================
# Step 2: Judge answers using API
# ============================================================

def get_token_provider():
    """Get Azure AD token provider."""
    from azure.identity import AzureCliCredential, get_bearer_token_provider
    if not TENANT_ID:
        raise ValueError("AZURE_TENANT_ID environment variable is required for Azure judge.")
    credential = AzureCliCredential(tenant_id=TENANT_ID)
    return get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )


def get_azure_client(token_provider):
    """Create an AzureOpenAI client with a randomly selected endpoint."""
    from openai import AzureOpenAI
    if not GPT4O_MINI_ENDPOINTS:
        raise ValueError(
            "No judge endpoints configured. Set AZURE_ENDPOINTS_FILE or "
            "JUDGE_API_BASE environment variable."
        )
    selected = random.choice(GPT4O_MINI_ENDPOINTS)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoint"],
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
        max_retries=3,
    )
    return client, selected["model"]


def judge_single_item(token_provider, item: Dict, image_base: str) -> Optional[Dict]:
    """Judge a single item using API."""
    image_path = os.path.join(image_base, item["image_url"])
    if not os.path.exists(image_path):
        return None

    question = item["question"]
    critical_points = item.get("critical_points", [])
    if not critical_points:
        return None

    thinking = item.get("thinking", "")
    summary = item.get("summary", "")
    answer_text = (thinking + "\n" + summary) if thinking else summary

    # Build judge prompt
    critical_points_text = "\n".join(critical_points)
    eval_prompt = f"""请根据问题、得分点列表和图像内容，对下面的回答进行评分。

问题：{question}

得分点列表：
{critical_points_text}

待评估回答：{answer_text}

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

    b64_image, mime_type = encode_image(image_path)
    messages = [
        {"role": "system", "content": "你是一个城市规划评估专家。"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
            {"type": "text", "text": eval_prompt},
        ]}
    ]

    for attempt in range(3):
        try:
            client, model = get_azure_client(token_provider)
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=4096,
            )
            score_text = response.choices[0].message.content or ""

            # Parse score
            score_match = re.search(r"最终得分：\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                total = float(score_match.group(2))
                normalized = (score / total) * 2 if total > 0 else 0
            else:
                normalized = 0.0

            result = item.copy()
            result["score"] = normalized
            result["score_text"] = score_text
            return result

        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                result = item.copy()
                result["score"] = 0.0
                result["score_text"] = f"[ERROR] {e}"
                return result


def judge_all_answers(results: List[Dict], image_base: str,
                      max_workers: int = 16) -> List[Dict]:
    """Judge all answers using API."""
    token_provider = get_token_provider()

    judged = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_single_item, token_provider, item, image_base): i
            for i, item in enumerate(results)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            result = future.result()
            if result:
                judged.append(result)

    return judged


# ============================================================
# Summary
# ============================================================

def print_summary(results: List[Dict], model_name: str = "") -> Dict:
    """Print and return evaluation summary."""
    type_scores: Dict[str, List[float]] = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["score"])

    print(f"\n===== {model_name} EVALUATION SUMMARY =====")
    all_scores = []
    summary = {}

    for t, scores in sorted(type_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {t}: avg={avg:.3f}/2, count={len(scores)}")
        all_scores.extend(scores)
        summary[t] = {"avg": round(avg, 3), "count": len(scores)}

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n  Overall: avg={overall:.3f}/2, total={len(all_scores)} items")
        summary["overall"] = {"avg": round(overall, 3), "count": len(all_scores)}
    print("=" * 40)

    return summary


# ============================================================
# Main evaluation pipeline
# ============================================================

def evaluate_model(model_path: str, model_name: str,
                   dataset_path: str = DEFAULT_DATASET,
                   image_base: str = DEFAULT_IMAGE_BASE,
                   output_dir: str = None,
                   judge_workers: int = 16) -> Dict:
    """Full evaluation pipeline for a single model."""

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "planbench_results")
    os.makedirs(output_dir, exist_ok=True)

    # Check if already evaluated
    final_path = os.path.join(output_dir, f"{model_name}_judged.json")
    if os.path.exists(final_path):
        print(f"\n[{model_name}] Already evaluated, loading results...")
        with open(final_path) as f:
            results = json.load(f)
        return print_summary(results, model_name)

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter valid items
    valid = [item for item in dataset
             if os.path.exists(os.path.join(image_base, item["image_url"]))]
    print(f"Valid items: {len(valid)}/{len(dataset)}")

    # Check if answers already generated
    answers_path = os.path.join(output_dir, f"{model_name}_answers.json")
    if os.path.exists(answers_path):
        print(f"Loading pre-generated answers from {answers_path}")
        with open(answers_path) as f:
            answers = json.load(f)
    else:
        # Step 1: Start vLLM and generate answers
        vllm_proc = None
        try:
            vllm_proc = start_vllm_server(model_path, port=VLLM_PORT)
            answers = generate_all_answers(valid, image_base, port=VLLM_PORT)

            # Save answers
            with open(answers_path, "w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(answers)} answers to {answers_path}")

        finally:
            if vllm_proc:
                stop_vllm_server(vllm_proc)
                time.sleep(5)  # GPU cooldown

    # Step 2: Judge answers
    print(f"\nJudging {len(answers)} answers...")
    judged = judge_all_answers(answers, image_base, max_workers=judge_workers)

    # Save judged results
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(judged, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(judged)} judged results to {final_path}")

    # Print summary
    return print_summary(judged, model_name)


def run_all_models(dataset_path: str = DEFAULT_DATASET,
                   image_base: str = DEFAULT_IMAGE_BASE,
                   output_dir: str = None,
                   judge_workers: int = 16):
    """Evaluate all models sequentially."""

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "planbench_results")

    all_summaries = {}

    for model_cfg in ALL_MODELS:
        model_path = model_cfg["model_path"]
        model_name = model_cfg["model_name"]

        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name}: path not found: {model_path}")
            continue

        try:
            summary = evaluate_model(
                model_path=model_path,
                model_name=model_name,
                dataset_path=dataset_path,
                image_base=image_base,
                output_dir=output_dir,
                judge_workers=judge_workers,
            )
            all_summaries[model_name] = summary
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)

    # Collect all types
    all_types = set()
    for s in all_summaries.values():
        all_types.update(k for k in s.keys() if k != "overall")
    all_types = sorted(all_types)

    # Header
    header = f"{'Model':<25} | " + " | ".join(f"{t:>6}" for t in all_types) + " | Overall"
    print(header)
    print("-" * len(header))

    for model_name, summary in all_summaries.items():
        scores = []
        for t in all_types:
            if t in summary:
                scores.append(f"{summary[t]['avg']:6.3f}")
            else:
                scores.append(f"{'--':>6}")
        overall = summary.get("overall", {}).get("avg", 0)
        print(f"{model_name:<25} | " + " | ".join(scores) + f" | {overall:.3f}")

    # Save summary
    summary_path = os.path.join(output_dir, "planbench_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="PlanBench-V evaluation for local VLM models")
    parser.add_argument("--run-all", action="store_true", help="Evaluate all models")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--model-name", type=str, help="Name for this model")
    parser.add_argument("--model-type", type=str, help="Model type (qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--image-base", default=DEFAULT_IMAGE_BASE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--judge-workers", type=int, default=16)
    args = parser.parse_args()

    if args.run_all:
        run_all_models(
            dataset_path=args.dataset,
            image_base=args.image_base,
            output_dir=args.output_dir,
            judge_workers=args.judge_workers,
        )
    elif args.model_path and args.model_name:
        evaluate_model(
            model_path=args.model_path,
            model_name=args.model_name,
            dataset_path=args.dataset,
            image_base=args.image_base,
            output_dir=args.output_dir,
            judge_workers=args.judge_workers,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_planbench_eval.py --run-all")
        print("  python run_planbench_eval.py --model-path /path/to/model --model-name MyModel")


if __name__ == "__main__":
    main()
