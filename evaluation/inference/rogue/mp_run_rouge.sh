PROJ_PATH=/root/paddlejob/workspace/env_run/sunhaoran06
model_name=exp3_new_repeat_5-gram

# 使用 mapfile 或 readarray 来读取目录内容到数组中
mapfile -t steps < <(ls ${PROJ_PATH}/${model_name} | grep step)
declare -a gpus=(0 1 2 3 4 5 6 7)
echo ${steps[@]}
gpu_index=0
group_size=8

mkdir -p ${PROJ_PATH}/results/${model_name}
echo > ${PROJ_PATH}/results/${model_name}/output.json

for ((i=0; i<${#steps[@]}; ++i)); do
    echo ${PROJ_PATH}/${model_name}/${steps[$i]}
    echo ${gpus[$gpu_index]}
    python ${PROJ_PATH}/ds-chat/evaluation/inference/rogue/run_rouge.py \
        --model ${PROJ_PATH}/${model_name}/${steps[$i]} \
        --gpu ${gpus[$gpu_index]} 1>>${PROJ_PATH}/results/${model_name}/output.json &
    
    gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    if (( (i+1) % group_size == 0 )); then
        wait
    fi
    sleep 1s
done

wait
