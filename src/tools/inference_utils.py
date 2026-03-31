"""
Inference Utilities — High-performance async inference backend.

Supports two modes:
  A) OpenAI-compatible (local vLLM/SGLang or any OpenAI-compatible server)
  B) Azure OpenAI API with multi-endpoint load balancing

Provides batch inference with:
  - asyncio-based high concurrency (50-100+ concurrent requests)
  - Progress bars (tqdm)
  - Automatic retries with exponential backoff
  - Speed statistics (tokens/s, requests/s)
  - Backward-compatible synchronous API
"""

import asyncio
import base64
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

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

# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------
BACKEND_OPENAI = "openai"
BACKEND_AZURE = "azure"


# ---------------------------------------------------------------------------
# Azure endpoint registry
# ---------------------------------------------------------------------------
def _get_azure_endpoints() -> Dict[str, list]:
    """
    Return Azure OpenAI endpoint registry with speed weights.

    Endpoints are loaded from the AZURE_ENDPOINTS_FILE environment variable
    (path to a JSON file) or from AZURE_ENDPOINTS (inline JSON string).

    Expected format:
    {
        "gpt-4o": [
            {"endpoint": "https://YOUR_RESOURCE.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
            ...
        ],
        "gpt-4o-mini": [...]
    }

    If no config is provided, returns an empty registry.
    """
    # Try file-based config first
    config_path = os.environ.get("AZURE_ENDPOINTS_FILE")
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)

    # Try inline JSON
    inline = os.environ.get("AZURE_ENDPOINTS")
    if inline:
        return json.loads(inline)

    # Default: empty registry
    logger.warning(
        "No Azure endpoints configured. Set AZURE_ENDPOINTS_FILE or AZURE_ENDPOINTS "
        "environment variable. See docs/azure_endpoints_example.json for format."
    )
    return {}


# ---------------------------------------------------------------------------
# InferenceParams (backward-compatible dataclass)
# ---------------------------------------------------------------------------
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
    max_workers: int = 1  # kept for backward compat; ignored in async mode

    # New async settings
    max_concurrency: int = 64
    empty_response_retries: int = 2
    empty_response_backoff: float = 0.5
    max_retries: int = 3
    retry_backoff_base: float = 1.0

    # Backend selection: "openai" or "azure"
    backend: str = BACKEND_OPENAI

    # Azure-specific settings
    azure_model_family: str = "gpt-4o"
    azure_tenant_id: str = os.environ.get("AZURE_TENANT_ID", "")
    azure_api_version: str = "2024-12-01-preview"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    system_prompt: str,
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


def _build_messages(
    prompt: str,
    image_path: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Build OpenAI chat messages with image."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------
class _Stats:
    """Track inference statistics."""
    def __init__(self):
        self.start_time = time.monotonic()
        self.completed = 0
        self.failed = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def record(self, usage: Optional[Any] = None):
        self.completed += 1
        if usage:
            self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def record_failure(self):
        self.failed += 1

    def summary(self) -> str:
        elapsed = time.monotonic() - self.start_time
        rps = self.completed / elapsed if elapsed > 0 else 0
        tps = self.total_completion_tokens / elapsed if elapsed > 0 else 0
        return (
            f"Completed: {self.completed} | Failed: {self.failed} | "
            f"Time: {elapsed:.1f}s | "
            f"RPS: {rps:.1f} req/s | "
            f"TPS: {tps:.0f} tok/s | "
            f"Tokens: {self.total_tokens} (prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens})"
        )


# ---------------------------------------------------------------------------
# Backend A: OpenAI-compatible async client (for vLLM / SGLang / etc.)
# ---------------------------------------------------------------------------
async def _openai_infer_one(
    client: Any,  # openai.AsyncOpenAI
    model: str,
    messages: List[Dict],
    params: InferenceParams,
    semaphore: asyncio.Semaphore,
    stats: _Stats,
    idx: int,
) -> Tuple[int, str]:
    """Infer a single request using the OpenAI async client."""
    last_error = None
    for attempt in range(params.max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                )
            content = response.choices[0].message.content or ""
            usage = response.usage

            # Retry on empty response
            if not content.strip():
                if attempt < params.empty_response_retries:
                    await asyncio.sleep(params.empty_response_backoff * (attempt + 1))
                    continue
                stats.record(usage)
                return idx, content.strip()

            stats.record(usage)
            return idx, content

        except Exception as e:
            last_error = e
            if attempt < params.max_retries:
                wait = params.retry_backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"[idx={idx}] Attempt {attempt+1} failed: {e}. Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[idx={idx}] All {params.max_retries+1} attempts failed: {last_error}")
                stats.record_failure()
                return idx, ""


async def _openai_batch_infer(
    prompts: List[str],
    image_paths: List[str],
    params: InferenceParams,
) -> List[str]:
    """Run batch inference using OpenAI async client against local/remote servers."""
    from openai import AsyncOpenAI

    model_name = params.model or os.environ.get("VLLM_MODEL") or os.environ.get("MODEL")
    if not model_name:
        raise ValueError("Missing model name. Set VLLM_MODEL or pass model=...")

    api_key = params.api_key
    if api_key is None:
        api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    base_urls = _resolve_base_urls(params)
    if not base_urls:
        raise ValueError("No server URL provided.")

    # Create one async client per base_url
    clients = []
    for url in base_urls:
        normalized = _normalize_api_base(url)
        clients.append(AsyncOpenAI(base_url=normalized, api_key=api_key))

    n = len(prompts)
    semaphore = asyncio.Semaphore(params.max_concurrency)
    stats = _Stats()

    # Build all tasks
    tasks = []
    for i in range(n):
        client = clients[i % len(clients)]
        messages = _build_messages(prompts[i], image_paths[i], params.system_prompt)
        tasks.append(
            _openai_infer_one(client, model_name, messages, params, semaphore, stats, i)
        )

    # Run with progress bar
    results = [""] * n
    pbar = tqdm(total=n, desc="Inference (OpenAI)", unit="req")
    for coro in asyncio.as_completed(tasks):
        idx, content = await coro
        results[idx] = content
        pbar.update(1)
    pbar.close()

    logger.info(f"Inference stats: {stats.summary()}")

    # Close clients
    for c in clients:
        await c.close()

    return results


# ---------------------------------------------------------------------------
# Backend B: Azure OpenAI async client with multi-endpoint load balancing
# ---------------------------------------------------------------------------
async def _azure_infer_one(
    idx: int,
    prompt: str,
    image_path: str,
    params: InferenceParams,
    semaphore: asyncio.Semaphore,
    stats: _Stats,
    token_provider: Any,
    endpoints: list,
    endpoint_weights: List[float],
) -> Tuple[int, str]:
    """Infer a single request via Azure OpenAI with weighted endpoint selection."""
    from openai import AsyncAzureOpenAI

    messages = _build_messages(prompt, image_path, params.system_prompt)
    last_error = None

    for attempt in range(params.max_retries + 1):
        # Pick endpoint by weighted random
        chosen = random.choices(endpoints, weights=endpoint_weights, k=1)[0]
        try:
            client = AsyncAzureOpenAI(
                azure_endpoint=chosen["endpoint"],
                azure_ad_token_provider=token_provider,
                api_version=params.azure_api_version,
            )
            async with semaphore:
                response = await client.chat.completions.create(
                    model=chosen["model"],
                    messages=messages,
                    max_completion_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                )
            await client.close()

            content = response.choices[0].message.content or ""
            usage = response.usage

            if not content.strip():
                if attempt < params.empty_response_retries:
                    await asyncio.sleep(params.empty_response_backoff * (attempt + 1))
                    continue
                stats.record(usage)
                return idx, content.strip()

            stats.record(usage)
            return idx, content

        except Exception as e:
            last_error = e
            if attempt < params.max_retries:
                wait = params.retry_backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"[idx={idx}] Azure attempt {attempt+1} failed ({chosen['endpoint']}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"[idx={idx}] All Azure attempts failed: {last_error}")
                stats.record_failure()
                return idx, ""


async def _azure_batch_infer(
    prompts: List[str],
    image_paths: List[str],
    params: InferenceParams,
) -> List[str]:
    """Run batch inference using Azure OpenAI with multi-endpoint load balancing."""
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    if not params.azure_tenant_id:
        raise ValueError(
            "Azure tenant ID is required. Set AZURE_TENANT_ID environment variable "
            "or pass azure_tenant_id to InferenceParams."
        )

    # Get token provider (sync, but cached)
    credential = AzureCliCredential(tenant_id=params.azure_tenant_id)
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    # Get endpoints for the requested model family
    all_endpoints = _get_azure_endpoints()
    model_family = params.azure_model_family
    if model_family not in all_endpoints:
        raise ValueError(
            f"Unknown Azure model family '{model_family}'. "
            f"Available: {list(all_endpoints.keys())}"
        )
    endpoints = all_endpoints[model_family]
    endpoint_weights = [e["speed"] for e in endpoints]

    logger.info(
        f"Azure backend: {len(endpoints)} endpoints for '{model_family}', "
        f"max_concurrency={params.max_concurrency}"
    )

    n = len(prompts)
    semaphore = asyncio.Semaphore(params.max_concurrency)
    stats = _Stats()

    tasks = [
        _azure_infer_one(
            i, prompts[i], image_paths[i], params,
            semaphore, stats, token_provider, endpoints, endpoint_weights
        )
        for i in range(n)
    ]

    results = [""] * n
    pbar = tqdm(total=n, desc="Inference (Azure)", unit="req")
    for coro in asyncio.as_completed(tasks):
        idx, content = await coro
        results[idx] = content
        pbar.update(1)
    pbar.close()

    logger.info(f"Azure inference stats: {stats.summary()}")
    return results


# ---------------------------------------------------------------------------
# Unified async dispatcher
# ---------------------------------------------------------------------------
async def _async_batch_infer(
    prompts: List[str],
    image_paths: List[str],
    params: InferenceParams,
) -> List[str]:
    """Dispatch to the right backend based on params.backend."""
    if params.backend == BACKEND_AZURE:
        return await _azure_batch_infer(prompts, image_paths, params)
    else:
        return await _openai_batch_infer(prompts, image_paths, params)


def _run_async(coro):
    """Run an async coroutine, handling the case where an event loop may already be running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an existing event loop (e.g. Jupyter) — use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # Fallback: run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------
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
    inference_params: Optional[InferenceParams] = None,
) -> List[str]:
    """
    Perform batch image inference — backward-compatible API.

    Internally uses asyncio with high concurrency for maximum throughput.
    Supports both OpenAI-compatible servers and Azure OpenAI.
    """
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

    if not prompts:
        return []

    return _run_async(_async_batch_infer(prompts, image_paths, inference_params))


# ---------------------------------------------------------------------------
# Async public API (for advanced use)
# ---------------------------------------------------------------------------
async def async_image_inference(
    prompts: List[str],
    image_paths: List[str],
    inference_params: InferenceParams,
) -> List[str]:
    """
    Async version of parallel_image_inference for direct use in async code.
    """
    if len(prompts) != len(image_paths):
        raise ValueError("The number of prompts and image_paths must match.")
    if not prompts:
        return []
    return await _async_batch_infer(prompts, image_paths, inference_params)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
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

    parser = argparse.ArgumentParser(description="Quick inference test")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--prompt", default="Describe the image briefly.", help="Prompt to use")
    parser.add_argument("--backend", default="openai", choices=["openai", "azure"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    images = _load_images(args.image_dir, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    params = InferenceParams(
        backend=args.backend,
        max_concurrency=args.concurrency,
    )

    start = time.time()
    results = parallel_image_inference(
        [args.prompt] * len(images),
        images,
        inference_params=params,
    )
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Images: {len(images)}, Concurrency: {args.concurrency}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per image: {elapsed/len(images):.2f}s")
    for i, r in enumerate(results):
        preview = r[:80].replace("\n", " ") if r else "(empty)"
        print(f"  [{i}] {preview}...")
