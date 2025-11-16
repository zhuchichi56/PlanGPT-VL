#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export DS_SKIP_CUDA_CHECK=1
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


num_processes=$(echo "$DEVICE" | tr ',' '\n' | wc -l)
# gradient_accumulation_steps=$((total_batch_size / (batch_size * num_processes)))

# 随机port 
port=$(( ( RANDOM % 1000 )  + 10000 ))
echo "port: $port"
echo "num_processes: $num_processes"
# echo "gradient_accumulation_steps: $gradient_accumulation_steps"
# echo "batch_size: $batch_size"
# echo "total_batch_size: $total_batch_size"
echo MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH
echo DATASET: $DATASET
echo OUTPUT_DIR: $OUTPUT_DIR
echo TEMPLATE: $TEMPLATE
echo DEVICE: $DEVICE


# deepspeed --num_nodes ${num_nodes} --num_gpus ${num_processes} \
#     --module src.train \
#         --deepspeed examples/deepspeed/ds_z3_config.json \
#         --stage sft \
#         --model_name_or_path "$MODEL_NAME_OR_PATH" \
#         --do_train \
#         --dataset "$DATASET" \
#         --template "$TEMPLATE" \
#         --finetuning_type full \
#         --output_dir "$OUTPUT_DIR" \
#         --per_device_train_batch_size "$batch_size" \
#         --gradient_accumulation_steps "$gradient_accumulation_steps" \
#         --save_strategy epoch \
#         --cutoff_len 2048 \
#         --eval_strategy "no" \
#         --save_total_limit 2 \
#         --weight_decay 0. \
#         --bf16 True \
#         --tf32 True \
#         --warmup_ratio 0.03 \
#         --logging_steps 1 \
#         --learning_rate 2e-5 \
#         --num_train_epochs 3 \
#         --lr_scheduler_type cosine \
#         --plot_loss \
#         --report_to wandb 




# accelerate launch --multi_gpu --num_processes ${num_processes} \
#     --num_machines ${num_nodes} src/train.py \
#         --deepspeed examples/deepspeed/ds_z3_config.json \
#         --stage sft \
#         --model_name_or_path "$MODEL_NAME_OR_PATH" \
#         --do_train \
#         --dataset "$DATASET" \
#         --template "$TEMPLATE" \
#         --finetuning_type full \
#         --output_dir "$OUTPUT_DIR" \
#         --per_device_train_batch_size "$batch_size" \
#         --gradient_accumulation_steps "$gradient_accumulation_steps" \
#         --save_strategy epoch \
#         --cutoff_len 2048 \
#         --eval_strategy "no" \
#         --save_total_limit 2 \
#         --weight_decay 0. \
#         --bf16 True \
#         --warmup_ratio 0.03 \
#         --logging_steps 1 \
#         --learning_rate 2e-5 \
#         --num_train_epochs 3 \
#         --lr_scheduler_type cosine \
#         --plot_loss \
#         --report_to wandb 
#         # --tf32 True \
    

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    /home/zhe/LLaMA-Factory/src/llamafactory/launcher.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage sft \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --do_train \
        --dataset SELECTED_DATA \
        --template $TEMPLATE \
        --finetuning_type full \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --save_strategy no \
        --cutoff_len 2048 \
        --eval_strategy "no" \
        --save_total_limit 0 \
        --logging_steps 10 \
        --learning_rate 1e-5 \
        --num_train_epochs 10 \
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
        




    
