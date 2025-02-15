# [ICLR'25 | MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions](https://arxiv.org/pdf/2410.02743)

   <a href="https://huggingface.co/baidu" target="_blank">
      <img alt="Models" src="https://img.shields.io/badge/🤗-Models-blue" />
   </a>
  <a href="https://openreview.net/forum?id=WWXjMYZxfH" target="_blank">
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" />
   </a>
  <a href="https://iclr.cc/Conferences/2025" target="_blank">
      <img alt="ICLR 2025" src="https://img.shields.io/badge/Proceedings-ICLR2025-red" />
   </a>



The official repository which contains the source code for our paper [MA-RLHF](https://openreview.net/pdf?id=WWXjMYZxfH).


## 🔥 News
* **22 January, 2025:** 🎉 We release the official codebase ([`ernie-research/MA-RLHF`](https://github.com/ernie-research/MA-RLHF/)). Stay tuned!🔥
* **22 January, 2025:** 🎉 Our work has been accepted to [ICLR 2025](https://iclr.cc/Conferences/2025)! 


## Get Started

### Prerequisites

For Gemma-2B and Gemma-7B:
- Pytorch == 1.12.1
- Python version == 3.10.14
- deepspeed == 0.13.1
- transformers == 4.41.0

For Gemma-2-27B:
- Pytorch == 2.2.0
- transformers == 4.44.0

We also provide the environments used in our experiments in `requirements.txt`, which can be easily installed with
```bash
cd ma-rlhf
pip install -r requirements.txt
```

### Installation

1. Installing our implemented deepspeed-chat:
```bash
cd MA-RLHF/applications/DeepSpeed-Chat
pip install -e .
```
2. Installing constituent_treelib for parsing based termination (Option)
```bash
cd MA-RLHF/Constituent-Treelib
pip install -e .
```

## Dataset

We conduct experiments on four datasets: [OpenAI TL;DR](https://huggingface.co/datasets/openai/summarize_from_feedback), [Anthropic HH-RLHF](https://huggingface.co/datasets/Dahoas/full-hh-rlhf), [OpenAI WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons), and [APPS](https://huggingface.co/datasets/codeparrot/apps). 

When training on TL;DR, HH-RLHF, and WebGPT, we split each dataset into three parts to carry out Supervised Fine-Tuning (SFT), reward modeling, and Reinforcement Learning with Human Feedback (RLHF) fine-tuning. The data is partitioned as follows: 20% for SFT, 40% for reward modeling, and 40% for RLHF fine-tuning. This partitioning is automated using Deepspeed-Chat codes with a random seed set to 1234.

For the APPS dataset, we divide the data into two parts: 20% for SFT fine-tuning and 80% for RLHF fine-tuning.

## Training 

### SFT Training

```shell
cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning && bash training_scripts/run_gemma.sh
```

### Reward Modeling

```shell
cd applications/DeepSpeed-Chat/training/step2_reward_model_finetuning && bash training_scripts/run_gemma.sh
```

### PPO Training

```shell
cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && bash training_scripts/run_gemma.sh
```

We implement 4 macro action termination conditions in this code, the default is fixed n-gram based termination. This can be changed by passing `--termination_condition` argument chosen from `[fixed, randomized, parsing, ppl]`. The hyper-parameters involved in the termination conditions can be specified with `--ngram` for fixed n-gram based termination, `--repeat_times` for randomized n-gram based termination, and `--parsing_cutoff` for parsing based termination. 

If the termination condition is set to `fixed` and `--ngram=1`, the code will initiate vanilla PPO training instead of MA-PPO.

Note that when training Gemma-2-27B, the `--bf16` flag should be enabled.

### Hyper-Parameters

|     |    Hyper-Parameter   |              Gemma              |                                            |        | CodeGemma |        |
|-----|:--------------------:|:-------------------------------:|:------------------------------------------:|:------:|:---------:|:------:|
|     |                      |                2B               |                     7B                     |   27B  |     2B    |   7B   |
| SFT |      Batch size      |   64 for WebGPT 512 for others  |                     128                    |   128  |     16    |   32   |
|     |        Epochs        |                3                |          5 for WebGPT 1 for others         |    3   |     1     |    1   |
|     |     Learning rate    | 1e-4 for WebGPT 5e-5 for others |                    2e-5                    |  5e-6  |    5e-6   |  2e-6  |
|     |     LR scheduler     |              cosine             |                   cosine                   | cosine |   cosine  | cosine |
|     |     Warmup ratio     |               0.1               |                     0.1                    |   0.1  |     0     |    0   |
|  RM |      Batch size      |   32 for WebGPT 64 for others   | 128 for TL;DR 64 for HH-RLHF 32 for WebGPT |   128  |     -     |    -   |
|     |        Epochs        |                1                |                      1                     |    1   |     -     |    -   |
|     |     Learning rate    | 2e-5 for WebGPT 1e-5 for others |                    1e-6                    |  8e-6  |     -     |    -   |
|     |     LR scheduler     |              cosine             |                   cosine                   | cosine |     -     |    -   |
|     |     Warmup ratio     |               0.1               |                     0.1                    |   0.1  |     -     |    -   |
| PPO |      Batch size      |               256               |                     256                    |   256  |     16    |   16   |
|     | Policy learning rate |              1.5e-5             |                    1e-6                    |  7e-7  |    5e-7   |  5e-7  |
|     | Critic learning rate |              1.5e-5             |                    1e-6                    |  1e-6  |    5e-5   |  5e-5  |
|     |        Epochs        |                1                |                      1                     |    1   |     1     |    1   |
|     |      PPO epochs      |                1                |                      1                     |    1   |     1     |    1   |
|     |        Rollout       |                1                |                      1                     |    1   |     1     |    1   |
|     |      Clip ratio      |               0.2               |                     0.2                    |   0.2  |    0.2    |   0.2  |
|     |   $\lambda$ in GAE   |               0.95              |                    0.95                    |  0.95  |    0.95   |  0.95  |
|     |    $\gamma$ in GAE   |                1                |                      1                     |    1   |     1     |    1   |
|     |    KL coefficient    |               0.05              |       0.1 for WebGPT 0.05 for others       |   0.1  |    0.05   |  0.05  |
|     |   Max prompt length  |               512               |                     512                    |   512  |    600    |   600  |
|     |  Max response length |               512               |                     512                    |   512  |    512    |   512  |
|     |     Warmup steps     |               200               |                     200                    |    0   |     20    |   20   |
|     |      Temperature     |               0.8               |                     0.8                    |   0.8  |    1.0    |   1.0  |
|     |         Top-p        |               1.0               |                     1.0                    |   1.0  |    1.0    |   1.0  |
|     |         Top-k        |                50               |                     50                     |   50   |     5     |    5   |

## Inference

We provide a Python file for inference located at `MA-RLHF/inference/inference_with_rewards.py`. In this file, we randomly select 2000 instances from the validation sets (using seed 1234) for inference. The generated responses are then scored with the reward model.

You can run the inference script with the following command:
```bash
cd MA-RLHF
python inference/inference_with_rewards.py \
    --proj-path ${PROJ_PATH} \
    --dataset summarize \
    --model ${actor_model_path} \
    --reward-model ${reward_model_path} \
    --temperature 0.8 \
    --gpu 0 \
    --batch-size 16
```
The `--dataset` argument can be chosen from `[summarize, hh-rlhf, webgpt]`, corresponding to openai/summarize_from_feedback, Dahoas/full-hh-rlhf, and openai/webgpt_comparisons, respectively. The inference results will be saved in  `${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name}.jsonl`.

## Evaluation

### RM Scores

The RM scores have already been computed during the inference stage.

### GPT-4 Evaluation

For GPT-4 evaluations, we randomly select 50 instances from the inference results. This can be done with the following command:
```bash
python evaluation/tools/sample_from_dataset.py \
    --data-path ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name}.jsonl \
    --save-path ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name}-sampled.jsonl \
    --dataset summarize
```
Specifically, we select 50 instances for the summarization task based on the provided SubReddit information by passing the `--dataset` summarize argument for a comprehensive evaluation.

The GPT-4 evaluation results can be obtained using:
```bash
# TL;DR and HH-RLHF
python evaluation/gpt4-eval.py \
    --model_name_a ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_a}-sampled.jsonl \
    --model_name_b ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_b}-sampled.jsonl \
    --dataset summarize \
    --output ${PROJ_PATH}/results/sumamrize/${actor_model_name_a}-v.s.-${actor_model_name_b}.jsonl \
    --sk ${OPENAI_SK}

# WebGPT
python evaluation/gpt4-webgpt-eval.py \
    --model_name_a ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_a}-sampled.jsonl \
    --model_name_b ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_b}-sampled.jsonl \
    --dataset summarize \
    --output ${PROJ_PATH}/results/sumamrize/gpt4-${actor_model_name_a}-v.s.-${actor_model_name_b}.jsonl \
    --sk ${OPENAI_SK}
```

The human evaluation results can be obtained using:
```bash
python evaluation/human-eval.py \
    --model_name_a ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_a}-sampled.jsonl \
    --model_name_b ${PROJ_PATH}/results/${dataset}/temperature=${temperature}/${actor_model_name_b}-sampled.jsonl \
    --output ${PROJ_PATH}/results/${dataset}/human-${actor_model_name_a}-v.s.-${actor_model_name_b}.jsonl
```

### Citation

```
@inproceedings{chai2025marlhf,
   title={{MA}-{RLHF}: Reinforcement Learning from Human Feedback with Macro Actions},
   author={Yekun Chai and Haoran Sun and Huang Fang and Shuohuan Wang and Yu Sun and Hua Wu},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=WWXjMYZxfH}
}
```
