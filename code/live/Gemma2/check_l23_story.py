"""L23 re-check: what does the corrected map say about the heads whose
ablation IMPROVES factual retrieval (L23H0 +7pts, L23H2 +4pts)?

Pulls per-head routing selectivity, OV copy/transform character, and
program-type mix for L23, with L22/L24 as context.
"""
import json

d = json.load(open("analysis_outputs/analysis_20260610_233407.json"))

def show_layer(L):
    lr = d["layer_results"][str(L)]
    print(f"\n=== Layer {L} (SAE layer {lr['sae_layer_used']}, "
          f"sliding={lr['is_sliding_window']}) ===")
    rr0 = lr["routing_results"][0]
    wr0 = lr["writing_results"][0]
    keys_r = [k for k in rr0 if isinstance(rr0[k], (int, float))]
    keys_w = [k for k in wr0 if isinstance(wr0[k], (int, float))]
    print("routing numeric keys:", keys_r)
    print("writing numeric keys:", keys_w)
    for h in range(8):
        rr = lr["routing_results"][h]
        wr = lr["writing_results"][h]
        pr = lr["program_results"][h]
        sel = rr.get("selectivity", rr.get("top1_mass"))
        diag = rr.get("diag_mass")
        copy = wr.get("copy_score", wr.get("mean_copy_alignment"))
        trans = wr.get("transform_score")
        ptypes = pr.get("program_type_counts", pr.get("type_histogram"))
        print(f" H{h}: sel={sel} diag={diag} copy={copy} transform={trans}")
        if ptypes:
            print(f"     programs: {ptypes}")

show_layer(23)
show_layer(22)
show_layer(24)
