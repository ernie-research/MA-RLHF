# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

This repository contains the source code for reproducing the MA-RLHF (PPO). We implement MA-RLHF on the basic of DeepSpeed-Chat tookits.

## Training 

### SFT Training

```shell
cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning && bash training_scripts/run_gemma_2b.sh
```

### Reward Modeling

```shell
cd applications/DeepSpeed-Chat/training/step2_reward_model_finetuning && bash training_scripts/run_gemma_2b.sh
```

### PPO Training

```shell
cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning && bash training_scripts/run_gemma_2b.sh
```

We implemente 4 macro action termination conditions in this code, the default is fixed n-gram based termination. This can be changed by pass `--termination_condition` argument choiced from `[fixed, randomized, parsing, ppl]`. The hyper-parameters involved in the termination conditions can be specified with `--ngram` for fixed n-gram based termination, `--repeat_times` for randomized n-gram based termination, and `--parsing_cutoff` for parsing based termination. 