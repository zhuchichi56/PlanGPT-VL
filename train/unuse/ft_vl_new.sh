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
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
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
FINETUNING_TYPE=${FINETUNING_TYPE:-"full"}  # 默认为full，可选值: full, lora
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-"true"}
FREEZE_MULTI_MODAL_PROJECTOR=${FREEZE_MULTI_MODAL_PROJECTOR:-"true"}
FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-"false"}
LEARNING_RATE=${LEARNING_RATE:-"2e-5"}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}

# adjust batch size; 
GLOBAL_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=1  # 固定为1
CUTOFF_LEN=8192 
LORA_RANK=16


# compute gradient accumulation steps; 
num_processes=$(echo "$DEVICE" | tr ',' '\n' | wc -l)
gradient_accumulation_steps=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * num_processes)))

# Random port
port=$(( ( RANDOM % 1000 )  + 10000 ))
echo "port: $port"
echo "num_processes: $num_processes" 
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "DATASET: $DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEMPLATE: $TEMPLATE"
echo "DEVICE: $DEVICE"
echo "FINETUNING_TYPE: $FINETUNING_TYPE"
echo "FREEZE_VISION_TOWER: $FREEZE_VISION_TOWER"
echo "FREEZE_MULTI_MODAL_PROJECTOR: $FREEZE_MULTI_MODAL_PROJECTOR"
echo "FREEZE_LANGUAGE_MODEL: $FREEZE_LANGUAGE_MODEL"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"

# 配置lora参数
LORA_PARAMS=""
if [ "$FINETUNING_TYPE" = "lora" ]; then
    LORA_PARAMS="--lora_rank $LORA_RANK --lora_target all"
fi

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
        --finetuning_type $FINETUNING_TYPE \
        $LORA_PARAMS \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --save_strategy "epoch" \
        --eval_strategy "no" \
        --cutoff_len $CUTOFF_LEN \
        --image_max_pixels 262144 \
        --video_max_pixels 16384 \
        --logging_steps 10 \
        --lr_scheduler_type cosine \
        --bf16 true \
        --tf32 true \
        --warmup_ratio 0.1 \
        --plot_loss \
        --preprocessing_num_workers 16 \
        --ddp_timeout 180000000 \
        --freeze_vision_tower $FREEZE_VISION_TOWER \
        --freeze_multi_modal_projector $FREEZE_MULTI_MODAL_PROJECTOR \
        --freeze_language_model $FREEZE_LANGUAGE_MODEL \
        --save_total_limit 5
        # --per_device_eval_batch_size 1 \
    
