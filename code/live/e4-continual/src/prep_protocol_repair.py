"""Build train/validation packs for the Qwen E4 protocol repair.

The untouched half-B test sets remain exactly as in the original E4. This script
splits the existing half-A source-item pools before packing, so validation packs
share no source items with optimization packs.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

SEED = 20260712
VAL_FRAC = {"math": 0.10, "code": 0.20}
TARGET_WORDS = {"math": 300, "code": 190}


def split_items(items, val_frac, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_val = max(1, int(round(len(items) * val_frac)))
    val_idx = set(order[:n_val].tolist())
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    assert set(train).isdisjoint(val)
    return train, val


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")


def digest(items):
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def main():
    from prep_e4 import rebuild_pools
    from build_manifest import pack_items

    halves = rebuild_pools()
    data = ROOT / "data"
    manifest = {"seed": SEED, "val_frac": VAL_FRAC, "splits": {}}

    for offset, cls in enumerate(("math", "code")):
        train_items, val_items = split_items(
            halves[cls]["A_items"], VAL_FRAC[cls], SEED + offset
        )
        train_packs = pack_items(
            train_items, TARGET_WORDS[cls], random.Random(SEED + 10 + offset)
        )
        val_packs = pack_items(
            val_items, TARGET_WORDS[cls], random.Random(SEED + 20 + offset)
        )
        train_path = data / f"repair_train_{cls}.jsonl"
        val_path = data / f"repair_val_{cls}.jsonl"
        write_jsonl(train_path, train_packs)
        write_jsonl(val_path, val_packs)
        manifest["splits"][cls] = {
            "n_train_items": len(train_items),
            "n_val_items": len(val_items),
            "n_train_packs": len(train_packs),
            "n_val_packs": len(val_packs),
            "train_items_sha256": digest(train_items),
            "val_items_sha256": digest(val_items),
            "train_file": str(train_path.relative_to(ROOT)),
            "val_file": str(val_path.relative_to(ROOT)),
        }
        print(
            f"{cls}: {len(train_items)} train items -> {len(train_packs)} packs; "
            f"{len(val_items)} validation items -> {len(val_packs)} packs"
        )

    with open(data / "repair_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {data / 'repair_split_manifest.json'}")


if __name__ == "__main__":
    main()
