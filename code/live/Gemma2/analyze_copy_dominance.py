"""
Script to find the head with highest copy_dominance score
"""
import json
import numpy as np

# Load the JSON file
print("Loading JSON file...")
with open(r'C:\Users\shubh\Downloads\s2path\Gemma2\analysis_outputs\analysis_20260102_052006.json', 'r') as f:
    data = json.load(f)

# Collect all copy_dominance scores
all_scores = []

print("Analyzing layers...")
for layer_key, layer_data in data['layer_results'].items():
    # Check for writing_results
    if 'writing_results' in layer_data:
        for entry in layer_data['writing_results']:
            layer_idx = entry.get('layer_idx')
            query_head = entry.get('query_head')
            kv_group = entry.get('kv_group')
            metrics = entry.get('metrics', {})
            
            if 'copy_dominance' in metrics:
                copy_dom = metrics['copy_dominance']
                all_scores.append({
                    'layer_idx': layer_idx,
                    'query_head': query_head,
                    'kv_group': kv_group,
                    'copy_dominance': copy_dom
                })

# Output to file
with open('copy_dominance_results.txt', 'w') as f:
    if not all_scores:
        msg = "No copy_dominance scores found!\n"
        first_layer = list(data['layer_results'].values())[0]
        msg += f"\nKeys in layer: {list(first_layer.keys())}\n"
        f.write(msg)
        print(msg)
    else:
        # Calculate statistics
        scores = [s['copy_dominance'] for s in all_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Find the max
        max_entry = max(all_scores, key=lambda x: x['copy_dominance'])
        
        lines = []
        lines.append("=" * 60)
        lines.append("COPY DOMINANCE ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nTotal heads analyzed: {len(all_scores)}")
        lines.append(f"\nMean copy_dominance: {mean_score:.6f}")
        lines.append(f"Std dev copy_dominance: {std_score:.6f}")
        lines.append("\n" + "=" * 60)
        lines.append("HIGHEST COPY_DOMINANCE HEAD:")
        lines.append("=" * 60)
        lines.append(f"  Layer: {max_entry['layer_idx']}")
        lines.append(f"  Query Head: {max_entry['query_head']}")
        lines.append(f"  KV Group: {max_entry['kv_group']}")
        lines.append(f"  Copy Dominance: {max_entry['copy_dominance']:.6f}")
        
        # Also show top 10
        lines.append("\n" + "=" * 60)
        lines.append("TOP 10 HEADS BY COPY_DOMINANCE:")
        lines.append("=" * 60)
        sorted_scores = sorted(all_scores, key=lambda x: x['copy_dominance'], reverse=True)[:10]
        for i, entry in enumerate(sorted_scores, 1):
            lines.append(f"  {i}. Layer {entry['layer_idx']}, Head {entry['query_head']}, KV Group {entry['kv_group']}: {entry['copy_dominance']:.6f}")
        
        output = "\n".join(lines)
        f.write(output)
        print("Results written to copy_dominance_results.txt")

print("Done!")
