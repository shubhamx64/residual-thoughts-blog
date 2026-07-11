import json
from pathlib import Path

conds = []
for f in sorted(Path('intervention_results').glob('*.json')):
    data = json.load(open(f))
    for c in data['conditions']:
        conds.append(c['condition_name'])
        
print("All conditions:")
for c in sorted(set(conds)):
    print(f"  {c}")
