from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dschat.utils.data.raw_datasets import OpenAISummarizeDataset, DahoasFullhhrlhfDataset, OpenaiWebgptcomparisonsDataset
from dschat.utils.model.model_utils import create_critic_model
from tqdm import tqdm
import os
import json
import torch

def generate_response(model, tokenizer, temperature, inputs):
    generate_kwargs = {
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
    }
    generated = model.generate(**inputs, max_new_tokens=512, **generate_kwargs)
        
    outputs = [tokenizer.decode(o[inputs["input_ids"].size(1):], skip_special_tokens=True) for o in generated]
    return outputs
    
def get_reward_score(reward_model, tokenizer, prompts, responses, prompt_length):
    tokenizer.padding_side = 'right'
    inputs = tokenizer.batch_encode_plus([p + r + tokenizer.eos_token for p, r in zip(prompts, responses)], return_tensors="pt", truncation=True, max_length=1024, padding=True).to(reward_model.rwtransformer.device)
            
    reward_score = reward_model.forward_value(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], prompt_length=prompt_length)['chosen_end_scores']
    tokenizer.padding_side = 'left'
    return reward_score

def main(args):
    PROJ_PATH=args.proj_path

    if args.dataset == "summarize":
        raw_dataset = OpenAISummarizeDataset("", 1234, 0, "openai/summarize_from_feedback")
    elif args.dataset == "hh-rlhf":
        raw_dataset = DahoasFullhhrlhfDataset("", 1234, 0, "Dahoas/full-hh-rlhf")
    elif args.dataset == "webgpt":
        raw_dataset = OpenaiWebgptcomparisonsDataset("", 1234, 0, "openai/webgpt_comparisons")

    validation = raw_dataset.get_eval_data()
    if args.dataset != "webgpt":
        validation = validation.shuffle(seed=1234)
    prompts, chosens, rejecteds = [], [], []
    for sample in validation:
        prompts.append(raw_dataset.get_prompt(sample))
        chosens.append(raw_dataset.get_chosen(sample))
        rejecteds.append(raw_dataset.get_rejected(sample))
    num_samples = 2000
    prompts = prompts[:num_samples]
    chosens = chosens[:num_samples]
    rejecteds = rejecteds[:num_samples]
    batch_size = args.batch_size
    torch.manual_seed(1234)
    step = args.model.split('/')[-1]
    model_name = args.model.split('/')[-2]
    reward_name = args.reward_model.split('/')[-2]
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(f"{args.model}/actor", device_map=f'cuda:{args.gpu + 1}')
        model = AutoModelForCausalLM.from_pretrained(f"{args.model}/actor", device_map=f"cuda:{args.gpu + 1}", torch_dtype=torch.float16)
        reward_model = create_critic_model(args.reward_model, tokenizer, None, rlhf_training=True).to(torch.float16).cuda(int(args.gpu))
        inferences = []
        model.eval()
        reward_model.eval()
        for i in tqdm(range(0, len(prompts), batch_size)):
            inputs = tokenizer.batch_encode_plus(prompts[i: i + batch_size], return_tensors="pt", truncation=True, max_length=1024, padding=True).to(model.device)

            if args.ref_scores:
                chosen_reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], chosens[i: i + batch_size], inputs["input_ids"].shape[-1])
                rejected_reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], rejecteds[i: i + batch_size], inputs["input_ids"].shape[-1])
                
                for j in range(len(chosen_reward_score)):
                    inferences.append({"prompt": prompts[i + j], "response": chosens[i + j], "reward": chosen_reward_score[j].item(), "rejected_response": rejecteds[i + j], "rejected_reward": rejected_reward_score[j].item()})
            else:
                response = generate_response(model, tokenizer, args.temperature, inputs)
                reward_score = get_reward_score(reward_model, tokenizer, prompts[i: i + batch_size], response, 2)
            
                for j in range(len(response)):
                    inferences.append({"prompt": prompts[i + j], "response": response[j], "reward": reward_score[j].item()})

    os.makedirs(f"{PROJ_PATH}/results{args.dataset}/temperature={args.temperature}/{model_name}", exist_ok=True)
    with open(f'{PROJ_PATH}/results{args.dataset}/temperature={args.temperature}/{model_name}.jsonl', 'w') as f:
        for inference in inferences:
            f.write(json.dumps(inference) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reward-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="summarize")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ref-scores", action='store_true')
    args = parser.parse_args()
    main(args)