PROJ_PATH=/root

OUTPUT=${PROJ_PATH}/models/summarize/sft

mkdir -p $OUTPUT

deepspeed --master_port 1234 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path ${PROJ_PATH}/gemma-2b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 5e-5 \
   --weight_decay 0.01 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.1 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
