import re
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

BIN_RE = re.compile(
    r"^\s*(\d+\-\d+|\d+\+):\s*r_s=([+\-]?\d+\.\d+)\s+r_p=([+\-]?\d+\.\d+)\s+"
    r"\(n=([\d,]+),\s*σ_pred=([\d.]+),\s*σ_act=([\d.]+)\)\s*$"
)

HEAD_START_RE = re.compile(r"^Layer\s+(\d+),\s*Head\s+(\d+)\s*$")
SUMMARY_RE = re.compile(r"^\s*Overall Pearson r:\s*([+\-]?\d+\.\d+)\s*$")
SUMMARY_S_RE = re.compile(r"^\s*Overall Spearman r:\s*([+\-]?\d+\.\d+)\s*$")
TOTAL_RE = re.compile(r"^\s*Total pairs:\s*([\d,]+)\s*$")
LOCAL_RE = re.compile(r"^\s*Local\s+\(0-4\):\s*([+\-]?\d+\.\d+)\s*$")
MID_RE = re.compile(r"^\s*Mid\s+\(16-32\):\s*([+\-]?\d+\d*\.?\d*)\s*$")
LONG_RE = re.compile(r"^\s*Long\s+\(128-256\):\s*([+\-]?\d+\d*\.?\d*)\s*$")
STAB_RE = re.compile(r"^\s*Sign stability:\s*([\d.]+)%\s*$")
SKIP_RE = re.compile(r"^\s*Rows:\s*(\d+)\/(\d+)\s*\(([\d.]+)%\)\s*$")
EMPTY_RE = re.compile(r"^\s*Empty bins:\s*(.*)\s*$")

@dataclass
class BinStats:
    r_s: float
    r_p: float
    n: int
    sigma_pred: float
    sigma_act: float

@dataclass
class HeadStats:
    layer: int
    head: int
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    total_pairs: Optional[int] = None
    local_s: Optional[float] = None
    mid_s: Optional[float] = None
    long_s: Optional[float] = None
    sign_stability: Optional[float] = None  # percent
    skipped_rows: Optional[int] = None
    total_rows: Optional[int] = None
    skipped_pct: Optional[float] = None
    empty_bins: List[str] = field(default_factory=list)
    bins: Dict[str, BinStats] = field(default_factory=dict)

def parse_int_commas(x: str) -> int:
    return int(x.replace(",", "").strip())

def parse_report(path: str) -> List[HeadStats]:
    heads: List[HeadStats] = []
    cur: Optional[HeadStats] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            m = HEAD_START_RE.match(line)
            if m:
                if cur is not None:
                    heads.append(cur)
                cur = HeadStats(layer=int(m.group(1)), head=int(m.group(2)))
                continue

            if cur is None:
                continue

            m = SUMMARY_RE.match(line)
            if m:
                cur.pearson = float(m.group(1))
                continue

            m = SUMMARY_S_RE.match(line)
            if m:
                cur.spearman = float(m.group(1))
                continue

            m = TOTAL_RE.match(line)
            if m:
                cur.total_pairs = parse_int_commas(m.group(1))
                continue

            m = LOCAL_RE.match(line)
            if m:
                cur.local_s = float(m.group(1))
                continue

            m = MID_RE.match(line)
            if m:
                cur.mid_s = float(m.group(1))
                continue

            m = LONG_RE.match(line)
            if m:
                cur.long_s = float(m.group(1))
                continue

            m = STAB_RE.match(line)
            if m:
                cur.sign_stability = float(m.group(1))
                continue

            m = SKIP_RE.match(line)
            if m:
                cur.skipped_rows = int(m.group(1))
                cur.total_rows = int(m.group(2))
                cur.skipped_pct = float(m.group(3))
                continue

            m = EMPTY_RE.match(line)
            if m:
                raw = m.group(1).strip()
                if raw:
                    cur.empty_bins = [s.strip() for s in raw.split(",")]
                continue

            m = BIN_RE.match(line)
            if m:
                bin_name = m.group(1)
                cur.bins[bin_name] = BinStats(
                    r_s=float(m.group(2)),
                    r_p=float(m.group(3)),
                    n=parse_int_commas(m.group(4)),
                    sigma_pred=float(m.group(5)),
                    sigma_act=float(m.group(6)),
                )
                continue

    if cur is not None:
        heads.append(cur)
    return heads

def farthest_nonempty_bin(head: HeadStats) -> Optional[Tuple[str, BinStats]]:
    # pick the bin with largest start number among bins with n>0
    best = None
    best_start = -1
    for name, st in head.bins.items():
        if st.n <= 0:
            continue
        if name.endswith("+"):
            start = int(name[:-1])
        else:
            start = int(name.split("-")[0])
        if start > best_start:
            best_start = start
            best = (name, st)
    return best

def safe(x: Optional[float], default: float = float("-inf")) -> float:
    return x if x is not None else default

def print_table(title: str, rows: List[HeadStats], key_fn, topk: int):
    rows = sorted(rows, key=key_fn, reverse=True)[:topk]
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"{'Head':>6}  {'S_all':>7}  {'P_all':>7}  {'S_0-4':>7}  {'S_16-32':>8}  {'S_far':>7}  {'far_bin':>8}  {'stab%':>6}  {'skip%':>6}")
    for h in rows:
        far = farthest_nonempty_bin(h)
        far_s = far[1].r_s if far else 0.0
        far_name = far[0] if far else "-"
        print(
            f"L{h.layer:02d}H{h.head}  "
            f"{safe(h.spearman):7.4f}  {safe(h.pearson):7.4f}  "
            f"{safe(h.local_s, 0.0):7.4f}  {safe(h.mid_s, 0.0):8.4f}  "
            f"{far_s:7.4f}  {far_name:>8}  "
            f"{safe(h.sign_stability, 0.0):6.2f}  {safe(h.skipped_pct, 0.0):6.2f}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="Path to validation .txt report")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--min_stab", type=float, default=70.0)
    args = ap.parse_args()

    heads = parse_report(args.report)
    if not heads:
        raise SystemExit("No heads parsed. Check the regex or report format.")

    # Helper filters
    stable = [h for h in heads if (h.sign_stability or 0.0) >= args.min_stab]
    not_too_skippy = [h for h in heads if (h.skipped_pct or 0.0) <= 10.0]

    # 1) Global winners
    winners = [h for h in stable if h.spearman is not None]
    print_table(
        f"Top {args.topk} heads by Overall Spearman (stab >= {args.min_stab}%)",
        winners,
        key_fn=lambda h: safe(h.spearman),
        topk=args.topk,
    )

    # 2) Local routers
    local = [h for h in stable if h.local_s is not None]
    print_table(
        f"Top {args.topk} heads by Local Spearman (0-4) (stab >= {args.min_stab}%)",
        local,
        key_fn=lambda h: safe(h.local_s, 0.0),
        topk=args.topk,
    )

    # 3) Mid-range routers
    mid = [h for h in stable if h.mid_s is not None]
    print_table(
        f"Top {args.topk} heads by Mid Spearman (16-32) (stab >= {args.min_stab}%)",
        mid,
        key_fn=lambda h: safe(h.mid_s, 0.0),
        topk=args.topk,
    )

    # 4) Long-ish range specialists (farthest non-empty bin)
    longish = [h for h in stable if farthest_nonempty_bin(h) is not None]
    print_table(
        f"Top {args.topk} heads by Spearman in farthest non-empty distance bin (stab >= {args.min_stab}%)",
        longish,
        key_fn=lambda h: farthest_nonempty_bin(h)[1].r_s if farthest_nonempty_bin(h) else float("-inf"),
        topk=args.topk,
    )

    # 5) Paradox heads: good local but negative overall Pearson
    paradox = [
        h for h in not_too_skippy
        if (h.local_s or 0.0) >= 0.10 and (h.pearson or 0.0) <= -0.15
    ]
    print_table(
        f"Paradox heads (local_s>=0.10 but pearson<=-0.15), top {args.topk} by local_s",
        paradox,
        key_fn=lambda h: safe(h.local_s, 0.0),
        topk=args.topk,
    )

    # 6) Anti-heads: strongly negative Spearman overall
    anti = [h for h in heads if (h.spearman is not None and h.spearman <= -0.10)]
    anti = sorted(anti, key=lambda h: safe(h.spearman), reverse=False)[:args.topk]
    print("\n" + "=" * 90)
    print(f"Most negative Overall Spearman (bottom {args.topk})")
    print("=" * 90)
    for h in anti:
        print(f"L{h.layer:02d}H{h.head}  spearman={h.spearman:.4f}  pearson={h.pearson:.4f}  stab={h.sign_stability or 0.0:.2f}%")

    # 7) Per-layer best head by overall Spearman
    by_layer: Dict[int, List[HeadStats]] = {}
    for h in heads:
        by_layer.setdefault(h.layer, []).append(h)
    print("\n" + "=" * 90)
    print("Best head per layer (by Overall Spearman)")
    print("=" * 90)
    for layer in sorted(by_layer.keys()):
        best = max(by_layer[layer], key=lambda h: safe(h.spearman))
        far = farthest_nonempty_bin(best)
        far_s = far[1].r_s if far else 0.0
        far_name = far[0] if far else "-"
        print(
            f"Layer {layer:02d}: L{best.layer:02d}H{best.head} "
            f"S_all={best.spearman or 0.0:.4f} P_all={best.pearson or 0.0:.4f} "
            f"S_0-4={best.local_s or 0.0:.4f} S_16-32={best.mid_s or 0.0:.4f} "
            f"S_far={far_s:.4f}({far_name}) stab={best.sign_stability or 0.0:.2f}%"
        )

if __name__ == "__main__":
    main()
