#export MODEL_PATH="/data/llama/13B_hf/"
export MODEL_PATH="/data/granite/13B-2.5T"
export DATA_PATH="/data/suneja/sft_alpaca_data.json"
#export OUTPUT_PATH="/data/suneja/llama_fms_oss/xxx/"
export OUTPUT_PATH="/data/suneja/fms_ckpt"
export PYTHONPATH="/data/suneja/fms-hf-tuning/"
export WANDB_PROJECT="llama"

nohup torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--master_port=1234 \
tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--data_path $DATA_PATH \
--bf16 True \
--output_dir $OUTPUT_PATH \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_config tuning/config/fsdp_config_granite.json \
--include_tokens_per_second \
--packing False \
--response_template "\n### Response:" \
--dataset_text_field "output" \
>nohup.out &
#--fsdp_config tuning/config/fsdp_config.json \
#--neftune_noise_alpha=10 \
