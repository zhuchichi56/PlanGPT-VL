import os
import json
import torch
import argparse
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, init_distributed_environment
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import sys

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class ImageInferenceResponse(BaseModel):
    outputs: List[str]

app = FastAPI()
llm = None
processor = None

def prepare_batch_inputs(messages_list, processor):
    """Prepare batch inputs for VLM inference"""
    batch_inputs = []

    for messages in messages_list:
        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, _ = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_input = {
            "prompt": chat_prompt,
            "multi_modal_data": mm_data,
        }

        batch_inputs.append(llm_input)

    return batch_inputs

@app.post("/inference", response_model=ImageInferenceResponse)
async def inference(request: Dict[str, Any]):
    """Main inference endpoint supporting both single and multi-GPU setups"""
    global llm, processor

    try:
        messages_list = request.get("messages")
        prompts = request.get("prompts")
        image_paths = request.get("image_paths")
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.1)
        top_p = request.get("top_p", 0.9)

        # Handle legacy format (prompts + image_paths)
        if not messages_list and (prompts and image_paths):
            messages_list = []
            for prompt, image_path in zip(prompts, image_paths):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
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
                messages_list.append(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        batch_inputs = prepare_batch_inputs(messages_list, processor)
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        result_texts = [output.outputs[0].text for output in outputs]

        return ImageInferenceResponse(outputs=result_texts)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_dataset")
async def process_dataset(request: Dict[str, Any]):
    """Process entire dataset for batch inference"""
    try:
        from utils import load_jsonlines, write_jsonlines
    except ImportError:
        logger.error("utils module not found. Please ensure utils.py is available.")
        raise HTTPException(status_code=500, detail="utils module not available")

    input_file = request.get("input_file")
    output_file = request.get("output_file")
    instruction_field = request.get("instruction_field", "instruction")
    image_field = request.get("image_field", "image_path")
    max_tokens = request.get("max_tokens", 512)

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")

    if os.path.exists(output_file):
        return {"status": "skipped", "message": f"Output file already exists: {output_file}"}

    try:
        data = load_jsonlines(input_file)
        logger.info(f"Loaded {len(data)} examples from {input_file}")

        prompts = [entry.get(instruction_field, "What is in this image?") for entry in data]
        image_paths = [entry.get(image_field, "") for entry in data]

        inference_request = {
            "prompts": prompts,
            "image_paths": image_paths,
            "max_tokens": max_tokens
        }

        response = await inference(inference_request)

        for i, entry in enumerate(data):
            entry["response"] = response.outputs[i]

        write_jsonlines(data, output_file)

        return {
            "status": "success",
            "processed": len(data),
            "output_file": output_file
        }

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "device": "multi_gpu" if torch.cuda.device_count() > 1 else "single_gpu"
    }

@app.get("/server_info")
async def server_info():
    """Get server configuration information"""
    gpu_count = torch.cuda.device_count()
    return {
        "gpu_count": gpu_count,
        "model_loaded": llm is not None,
        "cuda_available": torch.cuda.is_available(),
        "tensor_parallel_size": os.environ.get("TENSOR_PARALLEL_SIZE", "1")
    }

def initialize_model(model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85):
    """Initialize model with either single GPU or tensor parallelism"""
    global llm, processor

    if tensor_parallel_size > 1:
        logger.info(f"Initializing model with {tensor_parallel_size} GPUs (tensor parallelism): {model_path}")
        # Set GPU visibility for tensor parallelism
        gpus = list(range(tensor_parallel_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    else:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        logger.info(f"Initializing model on GPU {gpu_id}: {model_path}")

    try:
        model_config = {
            "model": model_path,
            "tokenizer_mode": "auto",
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 2, "video": 0},
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": 32768,
            "disable_log_stats": True,
        }

        # Add tensor parallelism config if needed
        if tensor_parallel_size > 1:
            model_config.update({
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": "bfloat16",
                "max_num_batched_tokens": 32768,
                "max_num_seqs": 8,
            })

        llm = LLM(**model_config)

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        logger.info("Model and processor initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

def cleanup():
    """Clean up model resources"""
    global llm
    if llm is not None:
        logger.info("Cleaning up model resources...")
        del llm
        torch.cuda.empty_cache()
        try:
            destroy_distributed_environment()
        except:
            pass

def main():
    """Main function to start the unified server"""
    parser = argparse.ArgumentParser(description="Unified VLM inference server (supports both single and multi-GPU)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization (0.0-1.0)")

    args = parser.parse_args()

    # Validate tensor parallelism
    if args.tensor_parallel_size > torch.cuda.device_count():
        logger.error(f"Tensor parallel size {args.tensor_parallel_size} > available GPUs {torch.cuda.device_count()}")
        sys.exit(1)

    # Store configuration for server info endpoint
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)

    if not initialize_model(args.model_path, args.tensor_parallel_size, args.gpu_memory_utilization):
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)

    try:
        import uvicorn
        mode = "multi-GPU tensor parallelism" if args.tensor_parallel_size > 1 else "single GPU"
        logger.info(f"Starting server on port {args.port} with {mode}")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        cleanup()

if __name__ == "__main__":
    main()