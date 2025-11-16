#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))


DEVICES="0,1,2,3"
TOTAL_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=1
LORA_RANK=8
LORA_TARGET="all"
PREF_BETA=0.01
PREF_LOSS="sigmoid"
SAVE_TOTAL_LIMIT=1
LOGGING_STEPS=1
LEARNING_RATE=5.0e-7
NUM_TRAIN_EPOCHS=1
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
PREPROCESSING_WORKERS=12
MAX_SAMPLES=10000
DDP_TIMEOUT=180000000
SEED=42
# MAX_LENGTH=2048
# MAX_PROMPT_LENGTH=1800
CUTOFF_LEN=512
# max_length: 1024
# max_prompt_length: 512





PORT=$(( ( RANDOM % 1000 )  + 10000 ))



while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICES="$2"
            # 计算GPU数量
            NUM_GPUS=$(echo $DEVICES | tr -cd ',' | wc -c)
            NUM_GPUS=$((NUM_GPUS + 1))
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
        *)
            shift
            ;;
    esac
done



NUM_PROCESSES=$(echo "$DEVICES" | tr ',' '\n' | wc -l)
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_PROCESSES)))

echo "NUM_GPUS: $NUM_GPUS"
echo "PER_DEVICE_BATCH_SIZE: $PER_DEVICE_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo "CUDA_VISIBLE_DEVICES: $DEVICES"


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nnodes 1 --node_rank 0 --nproc_per_node $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $PORT \
    src/llamafactory/launcher.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage dpo \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --do_train \
        --dataset $DATASET \
        --template $TEMPLATE \
        --finetuning_type full \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --pref_beta $PREF_BETA \
        --pref_loss $PREF_LOSS \
        --cutoff_len $CUTOFF_LEN \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --logging_steps $LOGGING_STEPS \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --lr_scheduler_type $LR_SCHEDULER \
        --bf16 True \
        --gradient_checkpointing True \
        --warmup_ratio $WARMUP_RATIO \
        --plot_loss \
        --preprocessing_num_workers $PREPROCESSING_WORKERS \
        --max_samples $MAX_SAMPLES \
        --overwrite_cache True \
        --overwrite_output_dir True \
        --ddp_timeout $DDP_TIMEOUT \
        --seed $SEED

        # --max_length $MAX_LENGTH \
        # --max_prompt_length $MAX_PROMPT_LENGTH \
        # --lora_rank $LORA_RANK \
        # --lora_target $LORA_TARGET \