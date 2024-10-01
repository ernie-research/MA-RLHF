PROJ_PATH=/root/paddlejob/workspace/env_run/sunhaoran06
model_name=exp5_beta=0.01_dpo

# 使用 mapfile 或 readarray 来读取目录内容到数组中
# mapfile -t steps < <(ls ${PROJ_PATH}/${model_name} | grep step)
declare -a steps=(9286 8499 7499 6499 5499 4499 3499 2499)
declare -a gpus=(0 1 2 3 4 5 6 7)
echo ${steps[@]}
gpu_index=0
group_size=8

mkdir -p ${PROJ_PATH}/results/${model_name}

for ((i=0; i<${#steps[@]}; ++i)); do
    echo ${PROJ_PATH}/${model_name}/${steps[$i]}
    echo ${gpus[$gpu_index]}
    python ${PROJ_PATH}/ds-chat/evaluation/inference/scores/run_reward.py \
        --model ${PROJ_PATH}/${model_name}/step-${steps[$i]} \
        --reward-model ${PROJ_PATH}/step2_0526v1_sft_ep3_epoch1_lr1e-5_0527v1/epoch_0 \
        --gpu ${gpus[$gpu_index]} \
        --batch-size 16 &
    
    gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    if (( (i+1) % group_size == 0 )); then
        wait
    fi
    sleep 1s
done

wait
