import json
import random
import os
import pathlib
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-file", type=str, required=True)
parser.add_argument("--save-file", type=str, required=True)
parser.add_argument("--dataset", type=str, default="summarize")
args = parser.parse_args()

filepath = args.filepath

random.seed(42)
samples = []
with open(filepath, 'r') as f:
    for line in f:
        samples.append(json.loads(line.strip()))

if args.dataset == "summarize":
    post_class = {}
    for idx, sample in enumerate(samples):
        prompt = sample['prompt']
        if prompt.startswith("ARTICLE"):
            post_type = "article"
        else:
            post_type = re.search(r'(?<=Subreddit: r/)\w+', prompt).group(0)
        if post_type not in post_class:
            post_class[post_type] = []
        post_class[post_type].append(idx)

    sampled_idx = []
    for key in post_class.keys():
        sampled_item = min(5, len(post_class[key]))
        sampled_idx.extend(random.sample(post_class[key], sampled_item))
    
    random.shuffle(sampled_idx)
    if len(sampled_idx) < 50:
        sampled_idx.extend(random.sample([i for i in range(len(samples)) if i not in sampled_idx], 50 - len(sampled_idx)))
    elif len(sampled_idx) > 50:
        sampled_idx = sampled_idx[:50]
    
    sampled_samples = [samples[i] for i in sampled_idx]
else:
    sampled_idx = random.sample([i for i in range(len(samples))], 50)
    sampled_samples = [samples[i] for i in sampled_idx]

os.makedirs('.'.join(args.savepath.split('.')[:-1]), exist_ok=True)
with open(args.savepath, 'w') as f:
    for sample in sampled_samples:
        f.write(json.dumps(sample) + '\n')