# VLM Inference Server (vLLM + LiteLLM)

本项目统一使用 **vLLM 的 OpenAI 兼容服务模式** 提供推理服务，并通过 **LiteLLM** 客户端调用。

## 启动服务器

### 单GPU模式(每个GPU一个进程)
```bash
./start_vllm.sh --model_path /path/to/your/model --tensor_parallel_size 1 --gpu_ids "0,1,2,3" --port 8000
```
*这会在端口8000-8003启动4个单GPU vLLM服务*

### 多GPU张量并行模式(每2个GPU一个进程)
```bash
./start_vllm.sh --model_path /path/to/your/model --tensor_parallel_size 2 --gpu_ids "0,1,2,3" --port 8000
```
*这会在端口8000-8001启动2个多GPU vLLM服务*

### 大模型模式(8个GPU张量并行)
```bash
./start_vllm.sh --model_path /path/to/your/model --tensor_parallel_size 8 --gpu_ids "0,1,2,3,4,5,6,7" --port 8000
```
*这会在端口8000启动1个8GPU vLLM服务*

## 客户端配置

LiteLLM 调用时需要提供模型名与 base URL：

```bash
export VLLM_MODEL=/path/to/your/model
export VLLM_API_BASE=http://localhost:8000/v1
```

如需认证，可设置：

```bash
export LITELLM_API_KEY=your_key
```

## 说明

- `start_vllm.sh` 严格按照项目既有推理参数启动 vLLM（dtype/bfloat16、max_model_len=32768、limit_mm_per_prompt、max_num_batched_tokens、max_num_seqs 等）。
- 推理调用统一在 `src/common/inference_utils.py` 中通过 LiteLLM 完成。
