##export MODEL_PATH="/data/granite/13B-2.5T"
export MODEL_PATH="/data/suneja/fms_ckpt/granite-13b/checkpoint-1219/"
export DATA_PATH="NA"
export OUTPUT_PATH="NA"
export PYTHONPATH="/data/suneja/fms-hf-tuning/"

python \
tuning/generate.py \
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
#--fsdp_config tuning/config/fsdp_config.json \
#--neftune_noise_alpha=10 \
