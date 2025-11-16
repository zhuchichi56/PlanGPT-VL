import asyncio
import httpx
import requests
import json
import os
import random
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger
import argparse
import itertools

class VLMClient:
    """Client for VLM inference servers (supports both single and multi-GPU setups)"""

    def __init__(self, base_urls: List[str] = None, base_url: str = "http://localhost:8000"):
        # 支持单个URL或URL列表
        if base_urls:
            self.base_urls = base_urls
        else:
            self.base_urls = [base_url]

        self.timeout = 600.0
        self.current_url_index = 0  # 轮询索引
        self.healthy_urls = set()   # 健康的URL集合

    async def health_check(self, url: str = None) -> Dict[str, Any]:
        """Check server health and get server info"""
        if url is None:
            url = self.base_urls[0]  # 默认检查第一个URL

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                health_data = response.json()

                # Get additional server info
                info_response = await client.get(f"{url}/server_info")
                info_response.raise_for_status()
                info_data = info_response.json()

                result = {**health_data, **info_data, "url": url}
                self.healthy_urls.add(url)
                return result
        except Exception as e:
            logger.error(f"Health check failed for {url}: {str(e)}")
            self.healthy_urls.discard(url)
            return {"status": "unhealthy", "error": str(e), "url": url}

    async def check_all_servers_health(self) -> Dict[str, Any]:
        """Check health of all servers"""
        health_results = {}
        tasks = []

        for url in self.base_urls:
            task = self.health_check(url)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            url = self.base_urls[i]
            if isinstance(result, Exception):
                health_results[url] = {"status": "unhealthy", "error": str(result), "url": url}
            else:
                health_results[url] = result

        return health_results

    def get_next_available_url(self) -> str:
        """获取下一个可用的URL (轮询)"""
        healthy_list = list(self.healthy_urls)
        if not healthy_list:
            # 如果没有健康的服务器,使用所有服务器进行轮询
            healthy_list = self.base_urls

        if not healthy_list:
            raise RuntimeError("No servers available")

        # 轮询选择
        url = healthy_list[self.current_url_index % len(healthy_list)]
        self.current_url_index += 1
        return url

    async def inference(self, messages_list: List[List[Dict]],
                      max_tokens: int = 256, temperature: float = 0.1,
                      top_p: float = 0.9) -> List[str]:
        """Perform inference with prepared messages"""
        url = self.get_next_available_url()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{url}/inference", json={
                "messages": messages_list,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            })
            response.raise_for_status()
            return response.json()["outputs"]

    async def inference_with_prompts_images(self, prompts: List[str], image_paths: List[str],
                                          max_tokens: int = 256, temperature: float = 0.1,
                                          top_p: float = 0.9) -> List[str]:
        """Perform inference with prompts and image paths (legacy format)"""
        url = self.get_next_available_url()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{url}/inference", json={
                "prompts": prompts,
                "image_paths": image_paths,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            })
            response.raise_for_status()
            return response.json()["outputs"]

    async def process_dataset(self, input_file: str, output_file: str,
                            instruction_field: str = "instruction",
                            image_field: str = "image_path",
                            max_tokens: int = 512) -> Dict[str, Any]:
        """Process entire dataset on server"""
        async with httpx.AsyncClient(timeout=3600.0) as client:
            response = await client.post(f"{self.base_url}/process_dataset", json={
                "input_file": input_file,
                "output_file": output_file,
                "instruction_field": instruction_field,
                "image_field": image_field,
                "max_tokens": max_tokens
            })
            response.raise_for_status()
            return response.json()

def prepare_messages(prompts: List[str], image_paths: List[str],
                    system_prompt: str = "You are a helpful assistant.") -> List[List[Dict]]:
    """Prepare message structure for the vision model"""
    num_images = len(image_paths)
    assert num_images == len(prompts), "The number of images and prompts must be the same"

    message_list = []
    for prompt, image_path in zip(prompts, image_paths):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        message_list.append(messages)

    return message_list

async def parallel_image_inference(prompts: List[str], image_paths: List[str],
                                 max_tokens: int = 256, temperature: float = 0.1,
                                 top_p: float = 0.9, server_url: str = "http://localhost:8000",
                                 server_urls: List[str] = None, use_messages_format: bool = True) -> List[str]:
    """Perform parallel image inference with multi-server support"""

    # 支持单个URL或URL列表
    if server_urls:
        client = VLMClient(base_urls=server_urls)
    else:
        client = VLMClient(base_url=server_url)

    # Check all servers health first
    health_results = await client.check_all_servers_health()
    healthy_count = sum(1 for result in health_results.values() if result.get("status") == "healthy")

    if healthy_count == 0:
        logger.error(f"No healthy servers available. Health results: {health_results}")
        raise RuntimeError("No servers are available")

    logger.info(f"Found {healthy_count}/{len(client.base_urls)} healthy servers")
    for url, health in health_results.items():
        if health.get("status") == "healthy":
            logger.info(f"Healthy server: {url}")

    if use_messages_format:
        messages = prepare_messages(prompts, image_paths)
        return await client.inference(messages, max_tokens, temperature, top_p)
    else:
        return await client.inference_with_prompts_images(prompts, image_paths, max_tokens, temperature, top_p)

def load_sample_images(image_mapping_path: str, num_samples: int = 5) -> List[Dict]:
    """Load sample images from the image function mapping file"""
    try:
        with open(image_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter for images that exist
        valid_samples = []
        for item in data:
            image_path = item['image_path']
            if os.path.exists(image_path):
                valid_samples.append(item)
                if len(valid_samples) >= num_samples * 2:
                    break

        # Randomly select samples
        selected_samples = random.sample(valid_samples, min(num_samples, len(valid_samples)))
        return selected_samples

    except Exception as e:
        logger.error(f"Error loading image mapping: {str(e)}")
        return []

async def test_electric_power_images(base_url: str = "http://localhost:8000", num_samples: int = 3):
    """Test with electric power equipment images"""
    logger.info(f"Testing with electric power equipment images ({num_samples} samples)...")

    image_mapping_path = "/volume/pt-train/users/wzhang/ghchen/zh/valid_code/code4elecgpt-v/image_function_mapping.json"

    if not os.path.exists(image_mapping_path):
        logger.error(f"Image mapping file not found: {image_mapping_path}")
        return False

    samples = load_sample_images(image_mapping_path, num_samples=num_samples)

    if not samples:
        logger.error("No valid images found for testing")
        return False

    client = VLMClient(base_url)
    success_count = 0

    for i, sample in enumerate(samples, 1):
        image_path = sample['image_path']
        function = sample['function']

        logger.info(f"--- Test {i}: {function} ---")
        logger.info(f"Image: {os.path.basename(image_path)}")

        # Customized prompts for different types of equipment
        if "断线散股" in function:
            prompt = "请详细描述这张电力设备图片中看到的线路问题,包括具体的缺陷类型和可能的原因。"
        elif "散热器" in function and "漏油" in function:
            prompt = "请描述这张变压器散热器图片中的油泄漏情况,包括泄漏的位置、严重程度和可能的处理建议。"
        elif "绝缘子" in function:
            prompt = "请分析这张绝缘子图片的状态,识别是否有损坏、污秽或其他异常情况。"
        else:
            prompt = "请详细描述这张电力设备图片中看到的内容,包括设备类型、状态和任何异常情况。"

        try:
            results = await client.inference_with_prompts_images([prompt], [image_path], max_tokens=200)
            logger.info(f"Response: {results[0]}")
            success_count += 1
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")

        logger.info("")  # Add spacing

    logger.info(f"Electric power image tests: {success_count}/{len(samples)} successful")
    return success_count > 0

async def main():
    """Main function for testing and demonstration"""
    parser = argparse.ArgumentParser(description="Unified VLM Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--test_mode", type=str, choices=["health", "text", "batch", "images"],
                       default="health", help="Test mode")
    parser.add_argument("--image", type=str, help="Path to specific test image")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片中的电力设备和状态",
                       help="Test prompt")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to test")

    args = parser.parse_args()

    client = VLMClient(args.url)

    if args.test_mode == "health":
        logger.info("Testing server health...")
        health = await client.health_check()
        logger.info(f"Health status: {health}")

    elif args.test_mode == "text":
        logger.info("Testing text-only inference...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好,请简单介绍一下你自己,说明你擅长什么类型的任务。"}
        ]
        results = await client.inference([messages], max_tokens=200)
        logger.info(f"Response: {results[0]}")

    elif args.test_mode == "batch":
        logger.info("Testing batch inference...")
        test_requests = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        messages_list = [test_requests]
        results = await client.inference(messages_list, max_tokens=50)
        logger.info(f"Batch response: {results[0]}")

    elif args.test_mode == "images":
        if args.image:
            logger.info("Testing specific image...")
            if not os.path.exists(args.image):
                logger.error(f"Image not found: {args.image}")
                return

            results = await client.inference_with_prompts_images([args.prompt], [args.image], max_tokens=300)
            logger.info(f"Image response: {results[0]}")
        else:
            await test_electric_power_images(args.url, args.samples)

# Export function for easy import
def parallel_image_inference(prompts: List[str], image_paths: List[str],
                           max_tokens: int = 256, temperature: float = 0.1,
                           top_p: float = 0.9, server_url: str = "http://localhost:8000",
                           server_urls: List[str] = None, use_messages_format: bool = True) -> List[str]:
    """
    并行图像推理函数 - 可以直接导入使用 (支持多服务器负载均衡)

    Args:
        prompts: 提示词列表
        image_paths: 图像路径列表
        max_tokens: 最大token数
        temperature: 温度参数
        top_p: top_p参数
        server_url: 单个服务器URL (向后兼容)
        server_urls: 服务器URL列表 (支持负载均衡)
        use_messages_format: 是否使用messages格式

    Returns:
        推理结果列表

    Example:
        from client import parallel_image_inference

        # 单服务器
        prompts = ["描述这张图片", "这张图片有什么特别之处?"]
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        results = parallel_image_inference(prompts, image_paths)

        # 多服务器负载均衡
        server_urls = ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002"]
        results = parallel_image_inference(prompts, image_paths, server_urls=server_urls)
    """
    return asyncio.run(_parallel_image_inference_async(
        prompts, image_paths, max_tokens, temperature, top_p, server_url, server_urls, use_messages_format
    ))

# Async helper function
async def _parallel_image_inference_async(prompts: List[str], image_paths: List[str],
                                        max_tokens: int, temperature: float,
                                        top_p: float, server_url: str, server_urls: List[str],
                                        use_messages_format: bool) -> List[str]:
    # 支持单个URL或URL列表
    if server_urls:
        client = VLMClient(base_urls=server_urls)
    else:
        client = VLMClient(base_url=server_url)

    # 检查服务器健康状态
    health_results = await client.check_all_servers_health()
    healthy_count = sum(1 for result in health_results.values() if result.get("status") == "healthy")

    if healthy_count == 0:
        logger.error(f"No healthy servers available")
        raise RuntimeError("No servers are available")

    logger.info(f"Using {healthy_count}/{len(client.base_urls)} healthy servers")

    if use_messages_format:
        messages = prepare_messages(prompts, image_paths)
        return await client.inference(messages, max_tokens, temperature, top_p)
    else:
        return await client.inference_with_prompts_images(prompts, image_paths, max_tokens, temperature, top_p)

if __name__ == "__main__":
    asyncio.run(main())