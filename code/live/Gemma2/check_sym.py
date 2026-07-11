import json

with open('intervention_prompts_sym.json') as f:
    d = json.load(f)

orig = [p for p in d['prompts'] if p.get('order') == 'original']
flip = [p for p in d['prompts'] if p.get('order') == 'flipped']

print(f"Original: {len(orig)}, Flipped: {len(flip)}, Total: {len(d['prompts'])}")

# Check pairing
paired = 0
for p in flip:
    pair_id = p.get('pair_id')
    if pair_id is not None:
        paired += 1

print(f"Flipped prompts with pair_id: {paired}")

# Check A/B balance in original
a_correct = sum(1 for p in orig if p['correct_choice'] == 'A')
b_correct = sum(1 for p in orig if p['correct_choice'] == 'B')
print(f"Original A correct: {a_correct}, B correct: {b_correct}")

# In flipped, should be swapped
a_correct_flip = sum(1 for p in flip if p['correct_choice'] == 'A')
b_correct_flip = sum(1 for p in flip if p['correct_choice'] == 'B')
print(f"Flipped A correct: {a_correct_flip}, B correct: {b_correct_flip}")
