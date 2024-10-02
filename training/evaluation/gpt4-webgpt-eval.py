import json
from tqdm import tqdm
import time
import argparse
import random
from openai import OpenAI

def get_gpt_response(query, sk):
    client = OpenAI(api_key=sk)

    SUCCESS = False
    result = "None"
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=5
            )
            result = response
            response = result.choices[0].message.content
            SUCCESS = True
            break
        except Exception as e:
            print("error:", e)
            time.sleep(2)
            continue

    if SUCCESS:
        return response
    else:
        return "None"

WEBGPT_PROMPT_TEMPLATE = """
You will be given two answers written for an question. Your task is to pick the better one between them, based on the criteria.

Factual accuracy - which answer is more factually accurate?

Coherence - which answer is easier to follow?

Usefulness overall - all things considered, which answer would be more helpful to the person who asked this question?

You should output with a json format where the key is the criteria and the value is the choice you made, using 'A' stands for Answer A and 'B' stands for Answer B. If you think both answers are equally good, output 'E'. 

Question:

{question}

Answer A:

{answer_a}

Answer B:

{answer_b}

Your Judgement (you should also output the reason, note that you are allowed to think both responses
are equally good, then output with 'E'):

"""



def parse_response(resp, reverse_flag):
    if resp.startswith('```json'):
        resp = resp[7:]
    if resp.endswith('```'):
        resp = resp[:-3]
    resp = json.loads(resp)
    if reverse_flag:
        print(resp)
        for k, v in resp.items():
            if v == 'A':
                resp[k] = 'B'
            elif v == 'B':
                resp[k] = 'A'
        print(resp)
    return resp

def main(args):
    model_a_resp = []
    with open(args.model_name_a, "r") as f:
        for line in f:
            model_a_resp.append(json.loads(line.strip()))

    model_b_resp = []
    with open(args.model_name_b, "r") as f:
        for line in f:
            model_b_resp.append(json.loads(line.strip()))

    f = open(args.output, "a")
    
    start_offset = 0
    tbar = tqdm(range(len(model_a_resp[start_offset: 50])))
    
    for idx in tbar:
        idx = start_offset + idx
        prompt_a = model_a_resp[idx]['prompt']
        prompt_b = model_b_resp[idx]['prompt']
        assert prompt_a == prompt_b
        sum_a = model_a_resp[idx]['response']
        sum_b = model_b_resp[idx]['response']
        resp = {}
        reverse_flag = random.choice([True, False])
        winner = 'Tie'
        if reverse_flag:
            inst = WEBGPT_PROMPT_TEMPLATE.format(article=prompt_a, summary_a=sum_b, summary_b=sum_a)
            resp = get_gpt_response(inst, args.sk)
        else:
            inst = WEBGPT_PROMPT_TEMPLATE.format(article=prompt_a, summary_a=sum_a, summary_b=sum_b)
            resp = get_gpt_response(inst, args.sk)
        winner = {'resp': resp, 'reverse_flag': reverse_flag}
            
        new_prompt = {
            'prompt': prompt_a,
            f'{args.model_name_a}': sum_a,
            f'{args.model_name_b}': sum_b,
            f'{args.model_name_a}_reward_score': model_a_resp[idx]['reward'],
            f'{args.model_name_b}_reward_score': model_b_resp[idx]['reward'],
            'winner': winner
        }

        f.write(json.dumps(new_prompt) + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_a", type=str, required=True)
    parser.add_argument("--model_name_b", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sk", type=str, required=True)
    args = parser.parse_args()
    main(args)