import json

with open('intervention_prompts.json') as f:
    d = json.load(f)

print(f"Total prompts: {d['n_prompts']}")
print(f"Filler bins: {d['filler_word_counts']}")

# Check one prompt from each bin
for filler in d['filler_word_counts']:
    p = [p for p in d['prompts'] if p['filler_word_count'] == filler][0]
    print(f"{filler}w: prefix={len(p['prefix'])}, middle={len(p['middle'])}, suffix={len(p['suffix'])}, total={len(p['prompt_text'])}")

