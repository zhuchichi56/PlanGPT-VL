"""
Inference Utilities

Provides batch inference capabilities with checkpointing support.
"""

import base64
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional
from litellm import completion

warnings.filterwarnings(
    "ignore",
    message="PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings.*",
    category=UserWarning,
)


@dataclass
class InferenceParams:
    """Shared inference parameters for VLM calls."""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    server_url: str = "http://localhost:8000"
    server_urls: Optional[List[str]] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: str = "You are a helpful assistant."
    max_workers: int = 1
    empty_response_retries: int = 1
    empty_response_backoff: float = 0.5


def _guess_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def _image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        content = f.read()
    mime_type = _guess_mime_type(image_path)
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_api_base(api_base: str) -> str:
    if not api_base:
        return api_base
    normalized = api_base.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _resolve_params(
    inference_params: Optional[InferenceParams],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    server_url: str,
    server_urls: Optional[List[str]],
    model: Optional[str],
    api_key: Optional[str],
    system_prompt: str
) -> InferenceParams:
    if inference_params is not None:
        return inference_params
    return InferenceParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        server_url=server_url,
        server_urls=server_urls,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
    )


def _resolve_base_urls(params: InferenceParams) -> List[str]:
    env_api_base = os.environ.get("VLLM_API_BASE") or os.environ.get("LITELLM_API_BASE")
    if params.server_urls is None and env_api_base:
        return [env_api_base]
    return params.server_urls or [params.server_url]


def parallel_image_inference(
    prompts: List[str],
    image_paths: List[str],
    max_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    server_url: str = "http://localhost:8000",
    server_urls: Optional[List[str]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    inference_params: Optional[InferenceParams] = None
) -> List[str]:
    """Perform image inference via LiteLLM against a vLLM OpenAI-compatible server."""
    inference_params = _resolve_params(
        inference_params,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        server_url=server_url,
        server_urls=server_urls,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
    )

    if len(prompts) != len(image_paths):
        raise ValueError("The number of prompts and image_paths must match.")

    model_name = inference_params.model or os.environ.get("VLLM_MODEL") or os.environ.get("MODEL")
    if not model_name:
        raise ValueError("Missing model name. Set VLLM_MODEL or pass model=...")

    api_key = inference_params.api_key
    if api_key is None:
        api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    base_urls = _resolve_base_urls(inference_params)
    if not base_urls:
        raise ValueError("No server URL provided.")

    def run_single(idx: int, prompt: str, image_path: str) -> str:
        api_base = _normalize_api_base(base_urls[idx % len(base_urls)])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": inference_params.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        attempts = 0
        while True:
            response = completion(
                model=model_name,
                messages=messages,
                max_tokens=inference_params.max_tokens,
                temperature=inference_params.temperature,
                top_p=inference_params.top_p,
                api_base=api_base,
                api_key=api_key,
            )
            content = response["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            if attempts >= inference_params.empty_response_retries:
                return (content or "").strip()
            attempts += 1
            time.sleep(inference_params.empty_response_backoff)

    results: List[str] = [""] * len(prompts)
    if inference_params.max_workers <= 1:
        for idx, (prompt, image_path) in enumerate(zip(prompts, image_paths)):
            results[idx] = run_single(idx, prompt, image_path)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=inference_params.max_workers) as executor:
            future_to_idx = {
                executor.submit(run_single, idx, prompt, image_path): idx
                for idx, (prompt, image_path) in enumerate(zip(prompts, image_paths))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

    return results


def _load_images(image_dir: str, limit: int) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    images = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(exts)
    ]
    images.sort()
    return images[:limit] if limit else images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare concurrency timing")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--prompt", default="Describe the image briefly.", help="Prompt to use")
    args = parser.parse_args()

    images = _load_images(args.image_dir, 10)
    if len(images) < 10:
        raise SystemExit("Need at least 10 images to run the comparison.")

    start = time.time()
    parallel_image_inference(
        [args.prompt] * 10,
        images[:10],
        inference_params=InferenceParams(max_workers=10),
    )
    ten_worker_elapsed = time.time() - start

    start = time.time()
    parallel_image_inference(
        [args.prompt],
        [images[0]],
        inference_params=InferenceParams(max_workers=1),
    )
    one_worker_elapsed = time.time() - start

    print(f"10 workers / 10 images: {ten_worker_elapsed:.2f}s")
    print(f"1 worker / 1 image: {one_worker_elapsed:.2f}s")
