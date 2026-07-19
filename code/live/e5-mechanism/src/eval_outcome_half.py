"""Re-evaluate existing E4 final checkpoints on the PREREG outcome half of
eval_math (leakage-clean selector comparison, plan rev 3 test c).
Arms: baseline, weights, fisher x seeds 0-4 -> results/outcome_half_evals.json."""
import json

import torch

from common_m import OUTCOME_IDX, RESULTS, ckpt_B, eval_texts, load_model

MODEL_KEY = "tinyllama-1.1b"
ARMS = ("baseline", "weights", "fisher")
SEEDS = (0, 1, 2, 3, 4)


def main():
    from common_m import eval_nll
    model, tok = load_model(MODEL_KEY)
    texts = eval_texts("math", OUTCOME_IDX)
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            sd = torch.load(ckpt_B(MODEL_KEY, arm, seed), map_location="cuda")
            model.load_state_dict(sd, strict=False)
            nll, ppl = eval_nll(model, tok, texts)
            out[f"{arm}_s{seed}"] = {"nll_outcome": nll, "ppl_outcome": ppl}
            print(f"  {arm}_s{seed}: outcome-half NLL {nll:.4f}", flush=True)
    (RESULTS / "outcome_half_evals.json").write_text(json.dumps(out, indent=1))
    print("saved outcome_half_evals.json")


if __name__ == "__main__":
    main()
