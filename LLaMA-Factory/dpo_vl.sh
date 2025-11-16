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
        --finetuning_type)
            FINETUNING_TYPE="$2"
            shift 2
            ;;
        --freeze_vision_tower)
            FREEZE_VISION_TOWER="$2"
            shift 2
            ;;
        --freeze_multi_modal_projector)
            FREEZE_MULTI_MODAL_PROJECTOR="$2"
            shift 2
            ;;
        --freeze_language_model)
            FREEZE_LANGUAGE_MODEL="$2"
            shift 2
            ;;
        --global_batch_size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values if not provided
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"qwen/Qwen2.5-VL-7B-Instruct"}
DATASET=${DATASET:-"rlhf_v"}
OUTPUT_DIR=${OUTPUT_DIR:-"saves/qwen2_vl-7b/lora/dpo"}
TEMPLATE=${TEMPLATE:-"qwen2_vl"}
DEVICE=${DEVICE:-"0,1,2,3"}
FINETUNING_TYPE=${FINETUNING_TYPE:-"full"}  # 默认为lora，可选值: full, lora
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-"true"}
FREEZE_MULTI_MODAL_PROJECTOR=${FREEZE_MULTI_MODAL_PROJECTOR:-"true"}
FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-"false"}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}

# Calculate batch size parameters
per_device_batch_size=2  # 固定为2
num_processes=$(echo "$DEVICE" | tr ',' '\n' | wc -l)
gradient_accumulation_steps=$((GLOBAL_BATCH_SIZE / (per_device_batch_size * num_processes)))


# Random port
port=$(( ( RANDOM % 1000 )  + 10000 ))
echo "port: $port"
echo "num_processes: $num_processes" 
echo "per_device_batch_size: $per_device_batch_size"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"
echo "global_batch_size: $GLOBAL_BATCH_SIZE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "DATASET: $DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEMPLATE: $TEMPLATE"
echo "DEVICE: $DEVICE"
echo "FINETUNING_TYPE: $FINETUNING_TYPE"
echo "FREEZE_VISION_TOWER: $FREEZE_VISION_TOWER"
echo "FREEZE_MULTI_MODAL_PROJECTOR: $FREEZE_MULTI_MODAL_PROJECTOR" 
echo "FREEZE_LANGUAGE_MODEL: $FREEZE_LANGUAGE_MODEL"


# 配置lora参数
LORA_PARAMS=""
if [ "$FINETUNING_TYPE" = "lora" ]; then
    LORA_PARAMS="--lora_rank 8 --lora_target all"
fi

torchrun --nnodes 1 --node_rank 0 --nproc_per_node $num_processes \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    src/llamafactory/launcher.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage dpo \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --trust_remote_code true \
        --do_train \
        --dataset $DATASET \
        --template $TEMPLATE \
        --finetuning_type $FINETUNING_TYPE \
        $LORA_PARAMS \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_strategy "epoch" \
        --cutoff_len 8192 \
        --image_max_pixels 262144 \
        --video_max_pixels 16384 \
        --learning_rate 5e-6 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --lr_scheduler_type cosine \
        --bf16 true \
        --warmup_ratio 0.1 \
        --plot_loss \
        --preprocessing_num_workers 16 \
        --dataloader_num_workers 4 \
        --ddp_timeout 180000000 \
        --eval_strategy "no" \
        --pref_beta 0.1 \
        --pref_loss sigmoid \
        --freeze_vision_tower $FREEZE_VISION_TOWER \
        --freeze_multi_modal_projector $FREEZE_MULTI_MODAL_PROJECTOR \
        --freeze_language_model $FREEZE_LANGUAGE_MODEL
        # --per_device_eval_batch_size 1 \
        # --max_samples 1000 \
        # --overwrite_cache true \
        # --save_only_model false \
        # --report_to none \
        # --tf32 true \
    

    