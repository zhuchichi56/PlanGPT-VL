#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
BASE_PORT=8000
TENSOR_PARALLEL_SIZE=1
GPU_IDS="0,1,2,3"
GPU_MEMORY_UTILIZATION=0.85

usage() {
  cat <<EOF
Usage: $0 --model_path /path/to/model [--port 8000] [--tensor_parallel_size 1] [--gpu_ids "0,1,2,3"] [--gpu_memory_utilization 0.85]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --port)
      BASE_PORT="$2"
      shift 2
      ;;
    --tensor_parallel_size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --gpu_ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --gpu_memory_utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Missing --model_path"
  usage
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPU IDs provided."
  exit 1
fi

export TOKENIZERS_PARALLELISM="false"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

pids=()

start_server() {
  local port="$1"
  local tp_size="$2"
  local gpu_list="$3"

  export CUDA_VISIBLE_DEVICES="$gpu_list"

  vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$port" \
    --tokenizer-mode auto \
    --tensor-parallel-size "$tp_size" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt "image=2,video=0" \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-log-stats &

  pids+=("$!")
  echo "Started vLLM server on port $port (CUDA_VISIBLE_DEVICES=$gpu_list, TP=$tp_size, PID=${pids[-1]})"
}

if [[ "$TENSOR_PARALLEL_SIZE" -eq 1 ]]; then
  for i in "${!GPU_LIST[@]}"; do
    port=$((BASE_PORT + i))
    start_server "$port" 1 "${GPU_LIST[$i]}"
    sleep 5
  done
else
  total_gpus="${#GPU_LIST[@]}"
  server_idx=0
  start_idx=0
  while [[ "$start_idx" -lt "$total_gpus" ]]; do
    end_idx=$((start_idx + TENSOR_PARALLEL_SIZE))
    if [[ "$end_idx" -gt "$total_gpus" ]]; then
      end_idx="$total_gpus"
    fi
    tp_size=$((end_idx - start_idx))
    gpu_slice=("${GPU_LIST[@]:start_idx:tp_size}")
    gpu_list_str=$(IFS=','; echo "${gpu_slice[*]}")
    port=$((BASE_PORT + server_idx))
    start_server "$port" "$tp_size" "$gpu_list_str"
    server_idx=$((server_idx + 1))
    start_idx=$((start_idx + TENSOR_PARALLEL_SIZE))
    sleep 10
  done
fi

cleanup() {
  echo "Stopping all servers..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -INT "$pid"
    fi
  done
  wait
}

trap cleanup INT TERM
wait
