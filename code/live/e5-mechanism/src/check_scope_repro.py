"""E-M2 code-path check (PREREG v1): a seed-0 --mask-scope all run's step-0 eval row
must exactly reproduce the original weights-arm step-0 row (log_B_weights.jsonl).
Compares ppl_math/ppl_code/ppl_prose exactly; fp_drift to 1e-6."""
import json
import sys
from pathlib import Path

E4 = Path(__file__).resolve().parents[2] / "e4-continual"


def row0(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


ref = row0(E4 / "results" / "log_B_weights.jsonl")
new = row0(E4 / "results" / "log_B_weights_scope-repro.jsonl")
ok = True
for k in ("ppl_math", "ppl_code", "ppl_prose"):
    match = ref[k] == new[k]
    ok &= match
    print(f"{k}: ref {ref[k]:.6f} new {new[k]:.6f} {'OK' if match else 'MISMATCH'}")
d = abs(ref.get("fp_drift", 0) - new.get("fp_drift", 0))
print(f"fp_drift: |diff| {d:.2e} {'OK' if d < 1e-6 else 'MISMATCH'}")
ok &= d < 1e-6
print("REPRO CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
