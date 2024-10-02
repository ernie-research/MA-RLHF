import json
import random
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_a", type=str, required=True)
parser.add_argument("--model_b", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

random.seed(42)

model_a = args.model_a
model_b = args.model_b

scored_samples = []

model_a_samples = []
with open(model_a, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        model_a_samples.append(data)

model_b_samples = []
with open(model_b, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        model_b_samples.append(data)

idx = [i for i in range(len(model_a_samples))]

evaluated_samples = []
for i in idx[:50]:
    assert model_a_samples[i]['prompt'] == model_b_samples[i]['prompt']
    print(f"Sample {i + 1}")
    print("=" * 50)
    print(f"Prompt: {model_a_samples[i]['prompt']}")
    change_case = random.choice([True, False])
    winner = ""
    ref = model_a_samples[i]['response'].strip().replace('\n', ' ')
    comp = model_b_samples[i]['response'].strip().replace('\n', ' ')
    if change_case:
        print("=" * 50)
        print(f"Resp A: {ref}")
        print("=" * 50)
        print(f"Resp B: {comp}")
        print("=" * 50)
        choice = input("1 for A, 2 for B, 3 for equal, 4 for pass, other for exit: ")
        if choice == '1':
            winner = "model_a"
        elif choice == '2':
            winner = "model_b"
        elif choice == '3':
            winner = "equal"
        elif choice == '4':
            continue
        else:
            print("You typed: ", choice)
            break
    else:
        print("=" * 50)
        print(f"Resp A: {comp}")
        print("=" * 50)
        print(f"Resp B: {ref}")
        print("=" * 50)
        choice = input("1 for A, 2 for B, 3 for equal, 4 for pass, other for exit: ")
        if choice == '1':
            winner = "model_b"
        elif choice == '2':
            winner = "model_a"
        elif choice == '3':
            winner = "equal"
        elif choice == '4':
            continue
        else:
            print("You typed: ", choice)
            break
    
    evaluated_samples.append({
        "prompt": model_a_samples[i]['prompt'],
        "model_a resp": model_a_samples[i]['response'],
        "model_b resp": model_b_samples[i]['response'],
        "winner": winner,
        "reward scores": {
            "model_a score": model_a_samples[i]['reward'],
            "model_b score": model_b_samples[i]['reward']
        }
    })

with open(args.output, 'a') as f:
    for sample in evaluated_samples:
        f.write(json.dumps(sample) + '\n')
    
