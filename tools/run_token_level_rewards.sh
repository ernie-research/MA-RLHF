PROJ_PATH=/root/paddlejob/workspace/env_run/sunhaoran06

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 ${PROJ_PATH}/ds-chat/tools/token_level_rewards.py --actor_critic_model_path ${PROJ_PATH}/models/exp3_5-gram --ref ${PROJ_PATH}/models/sft_baseline/2 --reward ${PROJ_PATH}/models/reward_baseline/epoch_0 --per_device_generation_batch_size 1 --per_device_training_batch_size 1 --steps 2399,2499,2599,2699,2799,2899,2999,3099,3199 --ngram 5

# 99,499,999,1499,1999,2499,2999,3499,3999,4499,4643,199,299,399,599,699,799,899,1099,1199,1299,1399,1599,1699,1799,1899,2099,2199,2299,2399,2599,2699,2799,2899,3099,3199,3299,3399,3599,3699,3799,3899,4099,4199,4299,4399,4599