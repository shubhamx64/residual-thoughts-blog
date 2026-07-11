"""L23 head character under the corrected map vs ablation effects."""
import json

d = json.load(open("analysis_outputs/analysis_20260610_233407.json"))

ABL = {0: +0.062, 1: +0.001, 2: +0.032, 3: +0.006, 4: 0.0, 5: -0.006,
       6: -0.005, 7: -0.022}  # accuracy delta vs baseline 0.595 (800 prompts)

wr0 = d["layer_results"]["23"]["writing_results"][0]
def tree(x, pre="", depth=0):
    if depth > 1: return
    if isinstance(x, dict):
        for k, v in list(x.items())[:30]:
            print(pre + k, type(v).__name__,
                  v if isinstance(v, (int, float, str, bool)) else "")
            tree(v, pre + "  ", depth + 1)
print("--- writing_results[0] structure ---")
tree(wr0)

print("\n--- per-head summary, L23 ---")
for h in range(8):
    rr = d["layer_results"]["23"]["routing_results"][h]["metrics"]
    wr = d["layer_results"]["23"]["writing_results"][h]
    wm = wr.get("metrics", wr)
    pr = d["layer_results"]["23"]["program_results"][h]
    counts = pr.get("program_type_counts") or pr.get("type_counts") or {}
    print(f"H{h}: ablation_delta={ABL[h]:+.3f} archetype={rr['archetype'].split('.')[-1]:<10} "
          f"diag_dom={rr['diagonal_dominance']:.3f} diag_mean={rr['diagonal_mean']:+.2f} "
          f"top1={rr['top1_mass_mean']:.4f}")
    numeric = {k: round(v, 3) for k, v in wm.items() if isinstance(v, (int, float))}
    print(f"     writing: {numeric}")
    if counts:
        print(f"     programs: {counts}")
