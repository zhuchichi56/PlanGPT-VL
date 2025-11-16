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

# Calculate batch size parameters
total_batch_size=128
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
echo MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH
echo DATASET: $DATASET
echo OUTPUT_DIR: $OUTPUT_DIR
echo TEMPLATE: $TEMPLATE
echo DEVICE: $DEVICE


torchrun --nnodes 1 --node_rank 0 --nproc_per_node $num_processes \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    src/llamafactory/launcher.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage sft \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --do_train \
        --dataset SELECTED_DATA \
        --template $TEMPLATE \
        --finetuning_type full \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_strategy no \
        --cutoff_len 16384 \
        --eval_strategy "no" \
        --save_total_limit 0 \
        --logging_steps 10 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --lr_scheduler_type cosine \
        --bf16 True \
        --tf32 True \
        --warmup_ratio 0.03 \
        --plot_loss \
        --preprocessing_num_workers 16 \
        --max_samples 1000000 \
        --overwrite_cache True  \
        --ddp_timeout 180000000 \
        --seed 42


        # --overwrite_output_dir \
        # --full_determinism \
        




    
