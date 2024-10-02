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

SUMMARY_PROMPT_TEMPLATE = """
You will be given two summaries written for an article. Your task is to pick the better one between them, based on the four criteria.
Please make sure you read and understand these instructions very carefully. 

Relevance - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.

Coherence - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."

Consistency - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.

Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

You should output single character to indicate which summary you think is better. 'A' stands for Summary A and 'B' stands for Summary B. If you think both summaries are equally good, output 'E'.

Article:

{article}

Summary A:

{summary_a}

Summary B:

{summary_b}

Your Choice (only a single character, you are allowed to think both summaries are equal and output 'E'):

"""

HHRLHF_PROMPT_TEMPLATE = """
For the following query to a chatbot assistant, which response is more helpful?

First provide a one-sentence comparison of the two responses and explain which you feel is more helpful. Second, on a new line, state only `A' or `B' to indicate which response is more helpful. If they are equally good or bad, state `E'. Your response should use the json format, with ``comparison'' and ``choice'' as keys.

Query: {dialogue}

Response A: {resp_a}

Response B: {resp_b}

Your Judgement:
"""

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
    
    a_win = 0
    b_win = 0
    tie = 0
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
            if args.dataset == 'summarize':
                inst = SUMMARY_PROMPT_TEMPLATE.format(article=prompt_a, summary_a=sum_b, summary_b=sum_a)
            elif args.dataset == 'hh-rlhf':
                inst = HHRLHF_PROMPT_TEMPLATE.format(dialogue=prompt_a, resp_a=sum_b, resp_b=sum_a)
            resp = get_gpt_response(inst, args.sk)

            if args.dataset != 'summarize':
                if resp.startswith("```json"):
                    resp = resp[7:-3]
                resp = json.loads(resp)
                comment = resp['comparison']
                resp = resp['choice']
                
                
            if resp == 'A':
                b_win += 1
                winner = 'B'
            elif resp == 'B':
                a_win += 1
                winner = 'A'
            elif resp == 'E':
                tie += 1
                winner = 'Tie'
            else:
                winner = resp
        else:
            if args.dataset == 'summarize':
                inst = SUMMARY_PROMPT_TEMPLATE.format(article=prompt_a, summary_a=sum_a, summary_b=sum_b)
            elif args.dataset == 'hh-rlhf':
                inst = HHRLHF_PROMPT_TEMPLATE.format(dialogue=prompt_a, resp_a=sum_a, resp_b=sum_b)
            resp = get_gpt_response(inst, args.sk)
            
            if args.dataset != 'summarize':
                if resp.startswith("```json"):
                    resp = resp[7:-3]
                resp = json.loads(resp)
                comment = resp['comparison']
                resp = resp['choice']
            
            if resp == 'A':
                a_win += 1
                winner = 'A'
            elif resp == 'B':
                b_win += 1
                winner = 'B'
            elif resp == 'E':
                tie += 1
                winner = 'Tie'
            else:
                winner = resp
         
        if args.dataset == 'summarize':   
            new_prompt = {
                'prompt': prompt_a,
                f'{args.model_name_a}': sum_a,
                f'{args.model_name_b}': sum_b,
                f'{args.model_name_a}_reward_score': model_a_resp[idx]['reward'],
                f'{args.model_name_b}_reward_score': model_b_resp[idx]['reward'],
                'winner': winner
            }
        elif args.dataset == 'hh-rlhf':
            new_prompt = {
                'prompt': prompt_a,
                f'{args.model_name_a}': sum_a,
                f'{args.model_name_b}': sum_b,
                f'{args.model_name_a}_reward_score': model_a_resp[idx]['reward'],
                f'{args.model_name_b}_reward_score': model_b_resp[idx]['reward'],
                'winner': winner,
                'comment': comment
            }
        
        f.write(json.dumps(new_prompt) + "\n")
    print(f"Total: {len(model_a_resp)}, A win: {a_win}, B win: {b_win}, Tie: {tie}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_a", type=str, required=True)
    parser.add_argument("--model_name_b", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sk", type=str, required=True)
    args = parser.parse_args()
    main(args)
