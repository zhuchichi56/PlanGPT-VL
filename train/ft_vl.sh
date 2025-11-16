#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT="1231"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values if not provided
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"qwen/Qwen2-VL-7B-Instruct"}
DATASET=${DATASET:-"review_mllm"}
OUTPUT_DIR=${OUTPUT_DIR:-"/share/home/tj24147/data/vlm/review_model_v1"}
TEMPLATE=${TEMPLATE:-"qwen2_vl"}
DEVICE=${DEVICE:-"0,1,2,3"}

# Calculate batch size parameters
total_batch_size=8
per_device_batch_size=1
num_processes=$(echo "$DEVICE" | tr ',' '\n' | wc -l)
gradient_accumulation_steps=$((total_batch_size / (per_device_batch_size * num_processes)))

# Random port
port=$(( ( RANDOM % 1000 )  + 10000 ))
echo "port: $port"
echo "num_processes: $num_processes" 
echo "per_device_batch_size: $per_device_batch_size"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"
echo "total_batch_size: $total_batch_size"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "DATASET: $DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEMPLATE: $TEMPLATE"
echo "DEVICE: $DEVICE"



# ### model
# model_name_or_path: /share/home/u24147/.cache/modelscope/hub/models/Qwen/Qwen2-VL-7B-Instruct
# image_resolution: 262144
# video_resolution: 16384
# trust_remote_code: true

# ### method
# stage: sft
# do_train: true
# finetuning_type: full
# freeze_vision_tower: true  # choices: [true, false]
# freeze_multi_modal_projector: true  # choices: [true, false]
# train_mm_proj_only: false  # choices: [true, false]
# deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

# ### dataset
# dataset: mllm_demo
# template: qwen2_vl
# cutoff_len: 2048
# max_samples: 1000
# overwrite_cache: true
# preprocessing_num_workers: 16

# ### output
# output_dir: saves/qwen2_vl-7b/full/sft
# logging_steps: 10
# save_steps: 500
# plot_loss: true
# overwrite_output_dir: true

# ### train
# per_device_train_batch_size: 1
# gradient_accumulation_steps: 2
# learning_rate: 1.0e-5
# num_train_epochs: 3.0
# lr_scheduler_type: cosine
# warmup_ratio: 0.1
# bf16: true
# ddp_timeout: 180000000

# ### eval
# # val_size: 0.1
# # per_device_eval_batch_size: 1
# # eval_strategy: steps
# # eval_steps: 500


torchrun --nnodes 1 --node_rank 0 --nproc_per_node $num_processes \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    src/llamafactory/launcher.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage sft \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --trust_remote_code true \
        --do_train \
        --dataset $DATASET \
        --template $TEMPLATE \
        --finetuning_type full \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --per_device_eval_batch_size 1 \
        --save_steps 3125 \
        --eval_steps 15 \
        --cutoff_len 8192 \
        --image_max_pixels 262144 \
        --video_max_pixels 16384 \
        --logging_steps 10 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --lr_scheduler_type cosine \
        --bf16 true \
        --tf32 true \
        --warmup_ratio 0.1 \
        --plot_loss \
        --preprocessing_num_workers 16 \
        --ddp_timeout 180000000 \
        --eval_strategy "no" \
        --freeze_vision_tower true \
        --freeze_multi_modal_projector true \
        --freeze_language_model false 
         # --freeze_vision_tower true \
        # --freeze_multi_modal_projector false \
        # --freeze_language_model false \
        # --eval_strategy "steps" \
        # --overwrite_output_dir
        # --val_size 0.1 \
        
