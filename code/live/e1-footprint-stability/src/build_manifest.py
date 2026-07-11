"""Build the E1' workload manifest: five text classes, document-disjoint halves.

Classes (surface form x computation regime dissociation):
  math        GSM8K gold solutions, calculator annotations stripped (arithmetic chains)
  math_prose  GSM8K questions (word problems: English surface, quantitative content)
  code        MBPP + HumanEval canonical solutions (pure code)
  code_prose  MBPP task descriptions (English imperatives about computation, no code syntax)
  prose       WikiText-103 article bodies, digit-light filter (narrative/expository)

Short items are packed into ~TARGET_WORDS-word sequences. Packing pools are
split into halves A/B at the source-item level BEFORE packing, so no source
document contributes to both halves.
"""
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from common import ROOT, CLASSES, SEED

TARGET_WORDS = 300      # ~380-420 tokens for most tokenizers
MIN_WORDS = 60
# shorter packs where the source pool is small, so we get enough documents;
# sequence length becomes a (documented) covariate for these classes
CLASS_TARGET_WORDS = {"code_prose": 110, "code": 190}
PACKS_PER_CLASS = 200   # 100 per half where the source supports it
CALIB_PER_CLASS = 10    # reserved for threshold calibration, excluded from analysis


def strip_gsm8k_solution(ans):
    ans = re.sub(r"<<[^>]*>>", "", ans)
    ans = ans.replace("####", "The answer is")
    return ans.strip()


def digit_fraction(text):
    if not text:
        return 1.0
    return sum(c.isdigit() for c in text) / len(text)


def pack_items(items, target_words, rng, sep="\n\n"):
    """Greedily pack a shuffled pool of short texts into ~target_words docs."""
    items = items[:]
    rng.shuffle(items)
    packs, cur, cur_w = [], [], 0
    for it in items:
        w = len(it.split())
        cur.append(it)
        cur_w += w
        if cur_w >= target_words:
            packs.append(sep.join(cur))
            cur, cur_w = [], 0
    if cur and cur_w >= MIN_WORDS:
        packs.append(sep.join(cur))
    return packs


def wikitext_articles():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    articles, cur = [], []
    title_re = re.compile(r"^ = [^=].* = $")
    n_scanned = 0
    for row in ds:
        line = row["text"]
        n_scanned += 1
        if title_re.match(line.rstrip("\n")):
            if cur:
                articles.append("".join(cur))
            cur = []
        elif line.strip() and not line.strip().startswith("="):
            cur.append(line)
        if len(articles) >= 3000 or n_scanned > 400_000:
            break
    if cur:
        articles.append("".join(cur))
    out = []
    for a in articles:
        a = re.sub(r"\s+", " ", a).strip()
        words = a.split()
        if len(words) < MIN_WORDS:
            continue
        a = " ".join(words[: TARGET_WORDS + 60])
        if digit_fraction(a) > 0.02:  # drop stat-heavy articles
            continue
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train+test+validation+prompt")
    he = load_dataset("openai/openai_humaneval", split="test")

    gsm_q = [r["question"].strip() for r in gsm]
    gsm_a = [strip_gsm8k_solution(r["answer"]) for r in gsm]
    mbpp_prompt = [r["text"].strip() for r in mbpp]
    mbpp_code = [r["code"].strip() for r in mbpp]
    he_code = [(r["prompt"] + r["canonical_solution"]).strip() for r in he]
    wiki = wikitext_articles()

    sources = {
        "math": (gsm_a, True),          # (item pool, needs packing)
        "math_prose": (gsm_q, True),
        "code": (mbpp_code + he_code, True),
        "code_prose": (mbpp_prompt, True),
        "prose": (wiki, False),         # already ~target length
    }

    manifest, texts = [], []
    counts = {}
    for cls in CLASSES:
        pool, needs_pack = sources[cls]
        pool = [p for p in pool if p]
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        half_a = [pool[i] for i in idx[: len(idx) // 2]]
        half_b = [pool[i] for i in idx[len(idx) // 2:]]
        docs = {}
        if needs_pack:
            tw = CLASS_TARGET_WORDS.get(cls, TARGET_WORDS)
            docs["A"] = pack_items(half_a, tw, rng)
            docs["B"] = pack_items(half_b, tw, rng)
        else:
            docs["A"], docs["B"] = half_a, half_b
        n_half = min(len(docs["A"]), len(docs["B"]), PACKS_PER_CLASS // 2 + CALIB_PER_CLASS)
        n_calib_half = CALIB_PER_CLASS // 2
        for half in ("A", "B"):
            sel = docs[half][:n_half]
            for j, text in enumerate(sel):
                role = "calib" if j < n_calib_half else "main"
                sid = f"{cls}_{half}_{j:04d}"
                manifest.append({"id": sid, "class": cls, "half": half, "role": role,
                                 "n_words": len(text.split())})
                texts.append({"id": sid, "text": text})
        counts[cls] = {"per_half": n_half, "main_per_half": n_half - n_calib_half}

    with open(ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(ROOT / "texts.jsonl", "w", encoding="utf-8") as f:
        for rec in texts:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"total sequences: {len(manifest)}")


if __name__ == "__main__":
    main()
