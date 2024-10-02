PROJ_PATH=/root
OUTPUT=${PROJ_PATH}/models/summarize/ma_ppo_fixed=5

mkdir -p $OUTPUT

Actor_Lr=1.5e-5
Critic_Lr=1.5e-5

deepspeed --num_gpus 8 --master_port 1234 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --actor_model_name_or_path ${PROJ_PATH}/models/summarize/sft \
   --critic_model_name_or_path ${PROJ_PATH}/models/summarize/reward_model \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --warmup_ratio 0.1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --end_of_conversation_token "<eos>" \
   --actor_dropout 0.0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 2 \
   --critic_zero_stage 3 \
   --offload \
   --kl_ctl 0.05 \
   --temperature 0.8 \
   --gamma 1.0 \
   --lam 0.95 \
   --termination_condition fixed \
   --ngram 5 \
   --value_function equal \
   --output_dir $OUTPUT \
   --print_answers \
   --print_answers_interval 100 \
   --save_steps 500 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
    &> $OUTPUT/training.log
