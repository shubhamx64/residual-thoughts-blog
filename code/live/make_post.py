"""Generate a self-contained single-file HTML post summarizing E1-E4 + E-Q.

All charts are inline SVG computed from the result JSONs; no external assets.
Output: ledger_post.html at repo root.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]
MC = {"qwen2.5-1.5b": "#2a78d6", "gemma-2-2b": "#eda100",
      "pythia-1.4b": "#4a3aa7", "tinyllama-1.1b": "#1baf7a"}
MLABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "gemma-2-2b": "Gemma-2-2B",
          "pythia-1.4b": "Pythia-1.4B", "tinyllama-1.1b": "TinyLlama-1.1B"}
INK, MUTED, GRID = "#16202b", "#5c6b7a", "#d7dde2"


def j(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ---------------- SVG helpers ----------------

def _ticks(lo, hi, n=5):
    import math
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    out = []
    t = t0
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def line_chart(series, title, xlabel, ylabel, w=640, h=320, y0=None, y1=None,
               hlines=(), note=""):
    """series: list of dicts {label, color, xs, ys, dash?}. Returns SVG string."""
    ml, mr, mt, mb = 58, 16, 34, 46
    pw, ph = w - ml - mr, h - mt - mb
    all_x = [x for s in series for x in s["xs"]]
    all_y = [y for s in series for y in s["ys"]] + [v for v, _ in hlines]
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo = min(all_y) if y0 is None else y0
    y_hi = max(all_y) if y1 is None else y1
    pad = (y_hi - y_lo) * 0.08 or 1
    if y0 is None:
        y_lo -= pad
    if y1 is None:
        y_hi += pad

    def X(x):
        return ml + (x - x_lo) / (x_hi - x_lo or 1) * pw

    def Y(y):
        return mt + ph - (y - y_lo) / (y_hi - y_lo or 1) * ph

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:{w}px;width:100%;height:auto;font-family:inherit">']
    p.append(f'<text x="{ml}" y="18" font-size="13.5" font-weight="600" fill="{INK}">{title}</text>')
    for t in _ticks(y_lo, y_hi):
        y = Y(t)
        p.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{ml-7}" y="{y+3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{t:g}</text>')
    for t in _ticks(x_lo, x_hi, 6):
        x = X(t)
        p.append(f'<text x="{x:.1f}" y="{h-mb+16}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{t:g}</text>')
    for v, lab in hlines:
        y = Y(v)
        p.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{INK}" '
                 f'stroke-width="1" stroke-dasharray="5 4"/>')
        p.append(f'<text x="{w-mr-2}" y="{y-4:.1f}" font-size="10" fill="{INK}" text-anchor="end">{lab}</text>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(s["xs"], s["ys"]))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.2"{dash}/>')
    # legend row under title
    lx = ml
    for s in series:
        p.append(f'<rect x="{lx}" y="{mt-10}" width="14" height="3.5" fill="{s["color"]}"/>')
        est = 7 * len(s["label"]) + 24
        p.append(f'<text x="{lx+18}" y="{mt-5}" font-size="10.5" fill="{INK}">{s["label"]}</text>')
        lx += est
    p.append(f'<text x="{ml}" y="{h-8}" font-size="10.5" fill="{MUTED}">{xlabel}</text>')
    p.append(f'<text x="14" y="{mt+12}" font-size="10.5" fill="{MUTED}" '
             f'transform="rotate(-90 14 {mt+12})" text-anchor="end">{ylabel}</text>')
    if note:
        p.append(f'<text x="{w-mr}" y="{h-8}" font-size="10" fill="{MUTED}" text-anchor="end">{note}</text>')
    p.append("</svg>")
    return "".join(p)


def bar_chart(groups, series_labels, colors, title, ylabel, w=640, h=300,
              fmt="{:.2f}", ymax=None):
    """groups: list of (group_label, [v_per_series])."""
    ml, mr, mt, mb = 58, 16, 34, 40
    pw, ph = w - ml - mr, h - mt - mb
    n_g, n_s = len(groups), len(series_labels)
    vmax = ymax or max(v for _, vs in groups for v in vs) * 1.15
    gw = pw / n_g
    bw = gw * 0.72 / n_s

    def Y(v):
        return mt + ph - v / vmax * ph

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:{w}px;width:100%;height:auto;font-family:inherit">']
    p.append(f'<text x="{ml}" y="18" font-size="13.5" font-weight="600" fill="{INK}">{title}</text>')
    for t in _ticks(0, vmax):
        y = Y(t)
        p.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        p.append(f'<text x="{ml-7}" y="{y+3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{t:g}</text>')
    for gi, (glab, vs) in enumerate(groups):
        x0 = ml + gi * gw + gw * 0.14
        for si, v in enumerate(vs):
            x = x0 + si * bw
            y = Y(v)
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-3:.1f}" '
                     f'height="{mt+ph-y:.1f}" rx="3" fill="{colors[si]}"/>')
            p.append(f'<text x="{x+(bw-3)/2:.1f}" y="{y-4:.1f}" font-size="9.5" '
                     f'fill="{INK}" text-anchor="middle">{fmt.format(v)}</text>')
        p.append(f'<text x="{ml+gi*gw+gw/2:.1f}" y="{h-mb+16}" font-size="10.5" '
                 f'fill="{INK}" text-anchor="middle">{glab}</text>')
    lx = ml
    for si, lab in enumerate(series_labels):
        p.append(f'<rect x="{lx}" y="{mt-11}" width="10" height="10" rx="2" fill="{colors[si]}"/>')
        p.append(f'<text x="{lx+14}" y="{mt-2}" font-size="10.5" fill="{INK}">{lab}</text>')
        lx += 7 * len(lab) + 30
    p.append(f'<text x="14" y="{mt+12}" font-size="10.5" fill="{MUTED}" '
             f'transform="rotate(-90 14 {mt+12})" text-anchor="end">{ylabel}</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------- data assembly ----------------

e1 = {m: j(ROOT / "e1-footprint-stability" / "results" / m / "metrics_q99.0.json") for m in MODELS}
e1c = {m: j(ROOT / "e1-footprint-stability" / "results" / m / "clustering_q99.0.json") for m in MODELS}
e2 = {m: j(ROOT / "e2-welch-gain" / "results" / m / "e2_metrics.json") for m in MODELS}
e3 = {m: j(ROOT / "e3-sufficiency" / "results" / m / "e3_metrics.json") for m in MODELS}
an = {m: j(ROOT / "e3-sufficiency" / "results" / m / "anatomy.json") for m in MODELS}
cm = {m: j(ROOT / "e3-sufficiency" / "results" / m / "conflict_matrix_distinctive.json") for m in MODELS}
e4 = j(ROOT / "e4-continual" / "results" / "e4_metrics.json")
e4_logs = {a: jsonl(ROOT / "e4-continual" / "results" / f"log_B_{a}.jsonl")
           for a in ("baseline", "random", "weights", "join", "footprint", "fisher")}
eqQ = {m: j(ROOT / "eq-quant-budget" / "results" / m / "eq_results.json")
       for m in ("tinyllama-1.1b", "qwen2.5-1.5b")}
sota = {m: j(ROOT / "eq-quant-budget" / "results" / m / "sota_results.json")
        for m in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")}

# E1 placement chart (per-layer, code_prose on qwen) + net-swing bars
qp = e1["qwen2.5-1.5b"]["contrast_placement"]["code_prose"]
chart_e1_lines = line_chart(
    [{"label": "footprint: to code", "color": "#1baf7a",
      "xs": list(range(len(qp["per_layer_to_sibling"]))), "ys": qp["per_layer_to_sibling"]},
     {"label": "footprint: to prose", "color": "#eda100",
      "xs": list(range(len(qp["per_layer_to_prose"]))), "ys": qp["per_layer_to_prose"]}],
    "Qwen2.5-1.5B — where does English-about-code sit, layer by layer?",
    "layer", "centered cosine to neighbor",
    hlines=[(qp["token_to_sibling"], "token baseline: to code"),
            (qp["token_to_prose"], "token baseline: to prose")])

swing_groups = []
for m in MODELS:
    cp = e1[m]["contrast_placement"]
    vals = []
    for k in ("math_prose", "code_prose"):
        p = cp[k]
        vals.append((p["to_sibling"] - p["token_to_sibling"]) - (p["to_prose"] - p["token_to_prose"]))
    swing_groups.append((MLABEL[m].split("-")[0], vals))
chart_e1_swing = bar_chart(swing_groups, ["math-prose → math", "code-prose → code"],
                           ["#2a78d6", "#1baf7a"],
                           "Net footprint swing toward the computation sibling (beyond token statistics)",
                           "net swing (centered cos)")

# E2 packing depth
chart_e2 = line_chart(
    [{"label": MLABEL[m], "color": MC[m],
      "xs": list(range(e2[m]["n_layers"])),
      "ys": [p["mlp_write"]["fp_ratio"] for p in e2[m]["per_layer"]]} for m in MODELS],
    "MLP write-dictionary packing efficiency by depth (frame-potential ratio; 1 = tight frame)",
    "layer", "FP_min / FP", y0=0, y1=0.85)

# E3 lift by stratum
chart_e3 = line_chart(
    [{"label": MLABEL[m], "color": MC[m],
      "xs": [(s["lo"] + s["hi"]) / 2 for s in e3[m]["lift_by_geom_stratum"]],
      "ys": [s["median_lift"] for s in e3[m]["lift_by_geom_stratum"]]} for m in MODELS],
    "Co-activation lift vs geometric overlap of write directions",
    "|cos(w_i, w_j)| (stratum midpoint)", "median lift  P(i∧j)/P(i)P(j)",
    hlines=[(1.0, "independence")])

# E4 trajectories
arm_cols = {"baseline": "#e34948", "random": "#898781", "weights": "#eda100",
            "join": "#2a78d6", "footprint": "#1baf7a", "fisher": "#4a3aa7"}
chart_e4 = line_chart(
    [{"label": a, "color": arm_cols[a],
      "xs": [r["step"] for r in log], "ys": [r["ppl_math"] for r in log]}
     for a, log in e4_logs.items()],
    "Task-A (math) retention while training task B (code) — TinyLlama, 20% of MLP neurons protected",
    "phase-B step", "held-out math ppl")

# E-Q scaling bars
eq_groups = []
for m, lab in (("qwen2.5-1.5b", "1.5B"), ("qwen2.5-3b", "3B"), ("qwen2.5-7b", "7B")):
    r = sota[m]
    un, bf = r["gptq3"]["wikitext2"], r["bf16"]["wikitext2"]
    fp = r["gptq_fp3"]["wikitext2"]
    hd = r["gptq_hdiag3"]["wikitext2"]
    eq_groups.append((lab, [100 * (un - fp) / (un - bf), 100 * (un - hd) / (un - bf)]))
chart_eq = bar_chart(eq_groups, ["count footprint (how often)", "energy / H-diag (how hard)"],
                     ["#1baf7a", "#2a78d6"],
                     "Share of 3-bit damage recovered by protecting 1% of neurons (WikiText-2)",
                     "% of ppl gap recovered", fmt="{:.0f}%")


def conflict_table(m):
    d = cm[m]
    cls = [c.replace("_prose", "-pr") for c in d["classes"]]
    M = d["normalized_conflict_layer_avg"]
    lo, hi = 0.15, 0.62
    rows = [f'<table class="cmx"><caption>{MLABEL[m]}</caption><tr><th></th>'
            + "".join(f"<th>{c}</th>" for c in cls) + "</tr>"]
    for a, ca in enumerate(cls):
        cells = [f"<th>{ca}</th>"]
        for b in range(len(cls)):
            if a == b:
                cells.append('<td class="diag">–</td>')
            else:
                v = M[a][b]
                t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
                # light blue -> dark blue
                col = f"rgba(37,106,191,{0.10 + 0.85*t:.2f})"
                ink = "#fff" if t > 0.55 else INK
                cells.append(f'<td style="background:{col};color:{ink}">{v:.2f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "".join(rows)


conflict_html = '<div class="cmwrap">' + "".join(conflict_table(m) for m in MODELS) + "</div>"

# assorted numbers for text
n = {}
for m in MODELS:
    d = e1[m]
    marg = [p["margin_ccos"] for p in d["per_layer"]]
    ns = [p["noise_std"] for p in d["per_layer"]]
    n[m] = {
        "marg_med": sorted(marg)[len(marg) // 2],
        "sigma_x": min(mm / (3 * s + 1e-12) for mm, s in zip(marg, ns)),
        "acc5": d["classification"]["acc5"], "tok5": d["classification"]["token_acc5"],
        "hdb": e1c[m]["all5"]["hdbscan_n_clusters"],
        "ari": e1c[m]["all5"]["hdbscan_ari_clustered_only"],
    }

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Ledger, Measured — weight-space experiments E1–E4 + quantization</title>
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --line:#c6cfd6; --paper:#edf0f2; --card:#f8fafb;
  --blue:#2a78d6; --blue-soft:#dde7fa; --amber:#c46a10; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink);
  font-family:Georgia,'Times New Roman',serif; font-size:17.5px; line-height:1.65; }}
.wrap {{ max-width:960px; margin:0 auto; padding:0 22px 80px; }}
.measure {{ max-width:700px; }}
header {{ border-bottom:1px solid var(--line); padding:18px 0; margin-bottom:44px;
  font-family:ui-monospace,Consolas,monospace; font-size:13px; color:var(--muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
header b {{ color:var(--ink); }}
h1 {{ font-family:system-ui,'Segoe UI',sans-serif; font-weight:800; font-size:clamp(30px,5.5vw,50px);
  line-height:1.05; letter-spacing:-0.02em; max-width:820px; }}
.dek {{ font-size:20px; color:var(--muted); font-style:italic; max-width:680px; margin:18px 0 8px; }}
.dek b {{ color:var(--ink); font-style:normal; }}
.status {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden; margin:34px 0 8px; }}
.status div {{ background:var(--card); padding:13px 15px; }}
.status .k {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); display:block; }}
.status .v {{ font-family:system-ui,sans-serif; font-weight:700; font-size:15px; }}
.status .pass {{ color:#0a7a3d; }} .status .split {{ color:var(--amber); }}
section {{ padding-top:52px; }}
.eyebrow {{ font-family:ui-monospace,Consolas,monospace; font-size:12.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--blue); margin-bottom:8px; }}
h2 {{ font-family:system-ui,'Segoe UI',sans-serif; font-weight:800; font-size:clamp(23px,3.4vw,32px);
  letter-spacing:-.015em; line-height:1.12; margin-bottom:16px; max-width:760px; }}
h3 {{ font-family:system-ui,sans-serif; font-weight:700; font-size:18px; margin:26px 0 8px; }}
p {{ margin-bottom:16px; max-width:700px; }}
strong {{ font-weight:600; }}
.verdict {{ border-left:3px solid var(--blue); background:var(--card); padding:14px 20px;
  margin:20px 0 24px; max-width:700px; border-radius:0 6px 6px 0; }}
.verdict .tag {{ font-family:ui-monospace,Consolas,monospace; font-size:11.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--blue); }}
.neg {{ border-left-color:var(--amber); }} .neg .tag {{ color:var(--amber); }}
table {{ border-collapse:collapse; margin:22px 0; font-size:14.5px; width:100%;
  font-family:system-ui,'Segoe UI',sans-serif; }}
th {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); text-align:left; padding:8px 10px;
  border-bottom:2px solid var(--ink); }}
td {{ padding:9px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
td.b, tr.hl td {{ font-weight:650; }}
tr.hl td {{ background:var(--blue-soft); }}
.tblwrap {{ overflow-x:auto; }}
figure {{ margin:26px 0; background:var(--card); border:1px solid var(--line);
  border-radius:6px; padding:18px 16px 10px; }}
figcaption {{ font-size:13.5px; color:var(--muted); padding:8px 6px 4px; line-height:1.5; }}
.cmwrap {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; margin:22px 0; }}
table.cmx {{ font-size:12.5px; margin:0; }}
table.cmx caption {{ font-family:system-ui,sans-serif; font-weight:700; font-size:13px;
  text-align:left; padding-bottom:6px; }}
table.cmx th {{ border-bottom:1px solid var(--line); font-size:10px; padding:4px 6px; }}
table.cmx td {{ text-align:center; padding:7px 6px; border-bottom:none; border-radius:3px; }}
table.cmx td.diag {{ color:var(--muted); background:none; }}
.note {{ font-size:14.5px; color:var(--muted); }}
.pull {{ border-left:3px solid var(--blue); padding:4px 0 4px 20px; margin:26px 0;
  font-size:19px; font-style:italic; max-width:640px; }}
footer {{ margin-top:70px; border-top:1px solid var(--line); padding-top:22px;
  font-family:ui-monospace,Consolas,monospace; font-size:12.5px; color:var(--muted);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
code {{ font-family:ui-monospace,Consolas,monospace; font-size:.85em; background:var(--card);
  border:1px solid var(--line); border-radius:3px; padding:1px 5px; }}
</style></head><body><div class="wrap">

<header><div><b>residual thoughts</b> / working notes</div><div>July 2–3, 2026 · one RTX 4060 Ti · four model families</div></header>

<h1>The Ledger, Measured<br><span style="color:var(--muted);font-weight:700">two days of weight-space experiments, E1–E4 plus quantization</span></h1>
<p class="dek">The capacity-ledger program claims a model's weights expose enough structure — packing, readers, gain — to account for its own capacity. This page reports what happened when every layer of that claim was <b>tested against activations and interventions</b>, on Qwen2.5-1.5B, Gemma-2-2B, Pythia-1.4B, and TinyLlama-1.1B, then scaled to Qwen2.5-7B for quantization.</p>

<div class="status">
<div><span class="k">E1 · footprints</span><span class="v pass">PASS</span></div>
<div><span class="k">E2 · packing + gain</span><span class="v pass">delivered</span></div>
<div><span class="k">E3 · sufficiency</span><span class="v split">SPLIT</span></div>
<div><span class="k">E4 · causal arms</span><span class="v pass">confirmed</span></div>
<div><span class="k">E-Q · quantization</span><span class="v pass">to 7B scale</span></div>
</div>

<!-- ============ SETUP ============ -->
<section>
<div class="eyebrow">Setup</div>
<h2>Five text classes, built so surface and computation disagree</h2>
<p class="measure">The original three-regime design (math / code / prose) would pass trivially — different tokens fire different neurons. The fix: two <strong>contrast classes</strong> whose surface is English but whose subject is computation. Everything below runs on ~900 packed sequences, split into halves A/B at the source-item level, with a token-unigram baseline computed alongside every footprint metric.</p>
<div class="tblwrap"><table>
<tr><th>class</th><th>source</th><th>surface form</th><th>computation</th></tr>
<tr><td>math</td><td>GSM8K gold solutions</td><td>arithmetic</td><td>math</td></tr>
<tr><td>math-prose</td><td>GSM8K questions (word problems)</td><td>English</td><td>math-adjacent</td></tr>
<tr><td>code</td><td>MBPP + HumanEval solutions</td><td>code</td><td>code</td></tr>
<tr><td>code-prose</td><td>MBPP task descriptions</td><td>English</td><td>code-adjacent</td></tr>
<tr><td>prose</td><td>WikiText articles (digit-light)</td><td>English</td><td>—</td></tr>
</table></div>
<p class="measure note">Sensor: fraction of tokens each MLP neuron's post-activation (the gated product feeding the down-projection) exceeds its layer's 99th-percentile threshold, streamed, never storing raw activations. Plus residual-stream participation ratio and per-sequence token counts.</p>
</section>

<!-- ============ E1 ============ -->
<section>
<div class="eyebrow">E1 · Footprint stability</div>
<h2>Footprints are stable, separable — and they track computation, not tokens</h2>
<div class="verdict"><span class="tag">Verdict — PASS</span><br>
Within-regime half-vs-half similarity ≈ 0.99 at every layer; the within-vs-across margin beats the pre-registered 3σ noise criterion by <strong>31–144×</strong> at the <em>worst</em> layer of every model; label shuffling kills it; single sequences classify at 98–100%. The one caveat: the token-unigram baseline also classifies at ~100%, so the regime evidence lives in the placement geometry below, not in accuracy.</div>

<div class="tblwrap"><table>
<tr><th>model</th><th>median margin (centered cos)</th><th>min margin ÷ 3σ noise</th><th>per-seq acc (5-class)</th><th>token baseline</th><th>HDBSCAN (no k)</th></tr>
{"".join(f"<tr><td>{MLABEL[m]}</td><td>{n[m]['marg_med']:.3f}</td><td>{n[m]['sigma_x']:.0f}×</td><td>{n[m]['acc5']:.3f}</td><td>{n[m]['tok5']:.3f}</td><td>{n[m]['hdb']} clusters, ARI {n[m]['ari']:.2f}</td></tr>" for m in MODELS)}
</table></div>

<h3>The decisive readout: where do the contrast classes sit?</h3>
<p class="measure">Token statistics say MBPP task descriptions are <em>maximally unlike</em> code (cosine −0.48 to −0.51 — they are plain English). Their footprints say otherwise: by layer ~3 they sit near code and move away from prose, and the effect holds in all four families and at all three firing thresholds.</p>
<figure>{chart_e1_lines}<figcaption>Half-footprint similarity of <b>code-prose</b> to its two possible neighbors, per layer, with the token-unigram baselines dashed. Surface statistics predict the bottom dashed line; the footprint spends the whole network far above it.</figcaption></figure>
<figure>{chart_e1_swing}<figcaption>Net swing toward the computation sibling — (footprint − token baseline) similarity to sibling, minus the same for prose. Positive = footprints see “text about computation” as computation. All eight bars positive.</figcaption></figure>
<p class="measure"><strong>Ontology check.</strong> HDBSCAN with no k recovers the five human classes on three models. Pythia disagrees in one informative way: the four task-like classes each form a pure cluster while <strong>prose fragments</strong> (161/210 sequences labeled noise). Task-like regimes are tight; “prose” is diffuse background — a ledger keyed on “prose” as one regime would key on a non-regime.</p>
</section>

<!-- ============ E2 ============ -->
<section>
<div class="eyebrow">E2 · Welch-gap packing + gain maps</div>
<h2>Pure weight-space instruments: slack is real, the worst case is universal</h2>
<div class="verdict"><span class="tag">Verdict — instruments delivered, one null, one delayed win</span><br>
Bulk packing sits 3–8× above the Welch floor everywhere, but max coherence is 0.78–1.00 at <em>every depth of every model</em> — near-duplicate write directions are universal. Per-layer packing does <em>not</em> predict where footprints separate (r = −0.39 / −0.15 / +0.29 / −0.01): the E1-bridge is null. The gain map, however, later earned a causal validation in the quantization arc: allocating high bits to high-gain layers beats the inverse decisively on Qwen (6.26 vs 8.18 mean ppl at equal budget).</div>
<figure>{chart_e2}<figcaption>Packing efficiency of the MLP write dictionaries (γ-folded down-projection columns). Gemma-2's monotone rise (r = +0.90 with depth) is <b>not</b> universal — Qwen is U-shaped. The safe cross-model claim is only “late ≥ mid”.</figcaption></figure>
<div class="tblwrap"><table>
<tr><th>model</th><th>token dict: FP ratio / q99 overlap</th><th>mlp-write FP early → late</th><th>max coherence (worst pair)</th><th>g_mlp peak (heuristic gain)</th></tr>
{"".join(f"<tr><td>{MLABEL[m]}</td><td>{e2[m]['token_dict']['fp_ratio']:.3f} / {e2[m]['token_dict']['q99']:.2f}</td><td>{e2[m]['per_layer'][0]['mlp_write']['fp_ratio']:.2f} → {e2[m]['per_layer'][-2]['mlp_write']['fp_ratio']:.2f}</td><td>{max(p['mlp_write']['coherence_max'] for p in e2[m]['per_layer']):.2f}</td><td>{max(p['gain']['g_mlp'] for p in e2[m]['per_layer']):.0f} (L{max(range(len(e2[m]['per_layer'])), key=lambda i: e2[m]['per_layer'][i]['gain']['g_mlp'])})</td></tr>" for m in MODELS)}
</table></div>
<p class="measure"><strong>Token dictionaries pack by vocabulary size, not family.</strong> Pythia (50k) and TinyLlama (32k) land at nearly identical FP ≈ 0.51 despite unrelated training; Qwen (152k) and Gemma (256k) are 10–40× sloppier. The token frame is a semantic dictionary, not a well-packed one — Gram machinery on it must treat heavy off-diagonal mass as the normal case. Gain maps are edge-hot in every model (writes near the first and last layers echo loudest); attention query reads are 1–3% stable rank everywhere, so packing scores only mean something for dictionaries meant to span.</p>
</section>

<!-- ============ E3 ============ -->
<section>
<div class="eyebrow">E3 · The sufficiency test</div>
<h2>Crowded pairs co-fire. Reader structure closes the gap only on Gemma-2.</h2>
<div class="verdict neg"><span class="tag">Verdict — split, program-defining</span><br>
<strong>H1:</strong> co-activation <em>rises</em> with geometric overlap in all four families — crowded neuron pairs co-fire, they do not avoid each other. <strong>H2:</strong> reader structure is a sufficient statistic for co-use on Gemma-2 only (reader-set Jaccard ρ = 0.38, subsumes geometry); elsewhere it is mostly a proxy for geometry. Even the full static model explains ≤ 17% of co-activation variance. <em>The weights-first audit does not close without the (cheap) activation calibration.</em></div>
<p class="measure"><strong>What H1 is and is not.</strong> This is a neuron-granularity result, and the anatomy below shows much of the high-|cos| crowding is <em>functional circuitry</em> — sign-opposed opponent couples and duplicate pairs of the kind Gurnee et&nbsp;al.'s <em>Universal Neurons</em> found replicating across random seeds. Co-firing circuit couples are doing their job, not paying a superposition tax. So H1 does <em>not</em> per se contradict the feature-level price-of-packing story: neurons are the screen, not the movie. The architectural dissociation — gated MLPs split crowding ~50/50 between opponent and duplicate couples while Pythia's plain-GELU crowding is mostly duplication — is, to our knowledge, new. Whether semantic mixing concentrates in the <em>genuinely unrelated</em> collisions rather than the couples is exactly E5's question.</p>
<figure>{chart_e3}<figcaption>~500 MLP-neuron pairs per layer, stratified by write-direction |cos| (never by reader structure); joint firing measured over the full E1 corpus. Orthogonal pairs co-fire at or below independence; crowded pairs at 1.6–3.5× above it — the dangerous direction, everywhere.</figcaption></figure>
<div class="tblwrap"><table>
<tr><th>model</th><th>ρ(geometry, lift)</th><th>ρ(reader-jac, lift)</th><th>partial ρ(readers | geom)</th><th>readers ΔR² | geom</th><th>geom ΔR² | readers</th><th>H2</th></tr>
{"".join(f"<tr{' class=hl' if m=='gemma-2-2b' else ''}><td>{MLABEL[m]}</td><td>{e3[m]['rho_geom_lift'][0]:+.2f}</td><td>{e3[m]['rho_readerjac_lift'][0]:+.2f}</td><td>{e3[m]['partial_rho_readerjac'][0]:+.2f}</td><td>{e3[m]['r2']['readers_given_geom']:.3f}</td><td>{e3[m]['r2']['geom_given_readers']:.3f}</td><td>{e3[m]['h2_verdict']}</td></tr>" for m in MODELS)}
</table></div>

<h3>Anatomy of the crowding (E3b)</h3>
<div class="tblwrap"><table>
<tr><th>model</th><th>anti-parallel share of high-|cos| pairs</th><th>token-set overlap: crowded vs orthogonal pairs</th><th>co-firing ≥70% one regime</th></tr>
{"".join(f"<tr><td>{MLABEL[m]}</td><td>{sorted(s['frac_antiparallel'] for s in an[m]['signed'] if s['n_high']>=10)[len([s for s in an[m]['signed'] if s['n_high']>=10])//2]:.0%}</td><td>{sorted(t['high_tok_overlap_med'] for t in an[m]['token_overlap'])[len(an[m]['token_overlap'])//2]:.2f} vs 0.00</td><td>{sorted(r['frac_concentrated_70'] for r in an[m]['regime_locality'])[len(an[m]['regime_locality'])//2]:.0%}</td></tr>" for m in MODELS)}
</table></div>
<p class="measure">Half of the crowding in gated-MLP models is <strong>opponent circuitry</strong> (sign-opposed write pairs), not redundancy — Pythia's plain-GELU crowding is mostly plain duplication (18%). Crowded pairs write toward the same vocabulary: packing neighborhoods are meaning neighborhoods.</p>

<h3>The conflict matrix — the ledger object, prototyped</h3>
<p class="measure">conflict(c₁,c₂) = Σᵢⱼ rᵢ(c₁)rⱼ(c₂)⟨wᵢ,wⱼ⟩² factorizes as ⟨M₁,M₂⟩_F with M_c = W·diag(r_c)·Wᵀ — computable from weights plus one cheap capture. Below on the <em>distinctive</em> substrate (class rate minus grand mean, clipped at zero).</p>
<div class="pull">The program's founding note contained one hypothetical sentence a self-aware model might say: <b>“math and coding share a crowded neighborhood; storywriting lives in sparse suburbs.”</b> Written before any measurement, it is now a table — math ↔ math-prose is the hottest pair and code-prose ↔ prose the coolest, four families out of four.</div>
{conflict_html}
</section>

<!-- ============ E4 ============ -->
<section>
<div class="eyebrow">E4 · Causal arms</div>
<h2>The ledger protects — and the information hierarchy is clean</h2>
<div class="verdict"><span class="tag">Verdict — all pre-registered hypotheses confirmed, then sharpened</span><br>
Math → code sequential fine-tune on TinyLlama (MLP-only), freezing 20% of neurons per arm at equal budget. Pre-registered ordering held exactly; two post-hoc arms (footprint-only, Fisher) then re-attributed the effect: <strong>usage information does the protective work; geometry is a zero-data proxy for it; gradient importance wins outright when affordable.</strong> And protection was a <strong>Pareto improvement</strong>: every informed arm also <em>acquired code better</em> than baseline (step-100 code ppl 6.0–6.2 vs 7.6). Working hypothesis: freezing high-usage substrate deflects new learning into free capacity instead of letting it scribble over contested weights — directly testable once E5's per-neuron mixing census exists (did the protected arms' code learning land in low-entropy regions?).</div>
<figure>{chart_e4}<figcaption>Held-out math perplexity during code training. Catastrophic forgetting is large (+248% for baseline); the protected arms separate in the pre-registered order at every checkpoint. Protection was Pareto — the protected arms also <b>learned code better</b>.</figcaption></figure>
<div class="tblwrap"><table>
<tr><th>arm</th><th>information</th><th>math ppl @100 / final</th><th>degradation</th><th>drift @100</th></tr>
{"".join(f"<tr><td>{a}</td><td>{d}</td><td>{e4_logs[a][1]['ppl_math']:.2f} / {e4['arms'][a]['retained_ppl_final']:.2f}</td><td>+{e4['arms'][a]['retention_degradation_pct']:.0f}%</td><td>{e4['arms'][a]['fp_drift_100']:.4f}</td></tr>" for a, d in (("baseline","none"),("random","budget only"),("weights","geometry (zero data)"),("join","geometry × footprint"),("footprint","footprint only (forward)"),("fisher","grad² on task A (backward)")))}
</table></div>
<p class="measure"><strong>Drift canary:</strong> footprint drift of a 40-sequence math probe rank-orders the six arms' final retention <em>exactly</em>, from the step-100 measurement alone (an exact match over 6 arms; p ≈ 0.0014 under a random ordering) — a task-resolved forgetting alarm readable long before behavioral evals move. Scope condition from the reverse direction (code→math, confounded by its 89-pack corpus): drift measures <em>change</em>, not harm — it is only a forgetting signal against a healthy reference footprint.</p>
<div class="pull">The packing term is causally inert once usage is measured; weight geometry alone still recovers ≈⅔ of Fisher's protection at zero data cost. The honest framing is a cost ladder — none &lt; forward-only &lt; backward — with real protection bought at every rung.</div>
</section>

<!-- ============ E-Q ============ -->
<section>
<div class="eyebrow">E-Q · Quantization budget</div>
<h2>Same ledger, different actuator: precision instead of learning rate</h2>
<div class="verdict"><span class="tag">Verdict — transfers, with scale-dependent structure</span><br>
A from-scratch GPTQ backbone (Hessian error compensation, group-128, layer streaming) reproduces published w4g128 baselines up to Qwen2.5-7B on this 16 GB card. At 4-bit, salience allocation is saturated for everyone (≤ 0.16 ppl). At 3-bit it matters — and which statistic matters is a function of scale.</div>
<div class="tblwrap"><table>
<tr><th>WikiText-2 ppl</th><th>1.5B</th><th>3B</th><th>7B</th></tr>
<tr><td>bf16</td><td>{sota['qwen2.5-1.5b']['bf16']['wikitext2']:.2f}</td><td>{sota['qwen2.5-3b']['bf16']['wikitext2']:.2f}</td><td>{sota['qwen2.5-7b']['bf16']['wikitext2']:.2f}</td></tr>
<tr><td>GPTQ w4g128</td><td>{sota['qwen2.5-1.5b']['gptq4']['wikitext2']:.2f}</td><td>{sota['qwen2.5-3b']['gptq4']['wikitext2']:.2f}</td><td>{sota['qwen2.5-7b']['gptq4']['wikitext2']:.2f}</td></tr>
<tr><td>GPTQ w3g128</td><td>{sota['qwen2.5-1.5b']['gptq3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-3b']['gptq3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-7b']['gptq3']['wikitext2']:.2f}</td></tr>
<tr><td>w3 + count footprint 1%</td><td>{sota['qwen2.5-1.5b']['gptq_fp3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-3b']['gptq_fp3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-7b']['gptq_fp3']['wikitext2']:.2f}</td></tr>
<tr class="hl"><td>w3 + energy (H-diag) 1%</td><td>{sota['qwen2.5-1.5b']['gptq_hdiag3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-3b']['gptq_hdiag3']['wikitext2']:.2f}</td><td>{sota['qwen2.5-7b']['gptq_hdiag3']['wikitext2']:.2f}</td></tr>
</table></div>
<figure>{chart_eq}<figcaption>Recovery of 3-bit damage by which 1% of neurons is protected. At 1.5B the energy signal is ~4× more effective — small-model quant error is dominated by extreme-magnitude outlier channels that firing counts can't see. By 7B the gap closes: <b>how often converges to how hard</b>, and the cheap forward-only footprint nearly suffices at deployment scale.</figcaption></figure>
<p class="measure">Two structural findings along the way. <strong>(1)</strong> On TinyLlama, parallel duplicate pairs are true redundant backups — 2-bit-destroying <em>one</em> member is free, destroying <em>both</em> is catastrophic (code ppl 12 → 90) — while opponent pairs partially self-cancel. On Qwen the dissociation vanishes: the redundancy reading of duplication is family-dependent. <strong>(2)</strong> Bit-mixing across tiers loses to uniform whenever the low tier is past the convexity cliff, no matter how good the map; salient protection (tiny fraction high, rest at working precision) is the only winning shape found. The E2 gain map earned one causal win: on Qwen, high-bits-to-high-gain layers beats the inverse decisively (6.26 vs 8.18 mean ppl).</p>
</section>

<!-- ============ SYNTHESIS ============ -->
<section>
<div class="eyebrow">Synthesis</div>
<h2>What the program looks like after contact with the data</h2>
<p class="measure"><strong>1 — Weights alone give real, zero-data structure</strong>: candidate crowded regions, gain topography, and protection maps worth ≈⅔ of Fisher's effect. They do not give co-use, and the packing term adds nothing once usage is measured.</p>
<p class="measure"><strong>2 — The activation calibration is cheap and mandatory.</strong> A ~4-minute forward-only capture is regime-faithful for <em>read</em> text (E1), carries most of the causal protective information (E4), and — extended with a second moment — becomes the right quantization salience at scale (E-Q). The fingerprint family should carry both moments: <em>counts</em> for regime discrimination, <em>energy</em> for precision allocation. The reading-vs-generating skew has now been measured (B1, Qwen + Gemma-2): regime <em>identity</em> survives the model's own sampling — generated sequences classify to the correct reading centroid at 95–97% — but absolute geometry does not (generation centroids sit 4.5–186× the reading noise floor away, and the offset is diffuse, not a correctable rigid shift). The harness therefore keeps <em>separate reading and generating reference footprints per regime</em>. And the regime ontology itself sharpened (B2): heterogeneous prose splits perfectly by register in all four families (ARI 1.0 — fiction and news are tight regimes; WikiText alone is diffuse), so fingerprint stores key on discovered registers, never on the label “prose”.</p>
<p class="measure"><strong>3 — Backward-looking importance wins when affordable.</strong> Fisher beat all forward-only arms at equal budget. The ledger's niche is where gradients, labels, and replay don't exist — continuous operation — and the open question is whether geometry helps in the low-data interpolation regime.</p>
<p class="measure"><strong>4 — Gemma-2 is anomalously weight-legible.</strong> Three independent instances (layer-6 separation peak, monotone packing depth, reader sufficiency PASS). Every structure→function mapping tested — reader sufficiency, duplicate redundancy, salience concentration — came out family-dependent. Single-model interpretability claims, including Gemma-first ones, carry a quantified optimism bias now.</p>
<p class="measure"><strong>5 — The drift canary works</strong> (ρ = 0.994), costs a 40-sequence probe, and has one documented scope condition: it needs a healthy reference.</p>
<p class="measure note">Everything above ran on one consumer GPU in two days: ~900-sequence captures per model, minutes-to-an-hour per experiment, seeds fixed, every claim carrying its cross-family caveat. Code and per-model JSONs live in the repo (<code>e1-footprint-stability/ … eq-quant-budget/</code>), each with a verdict-first REPORT.md.</p>
</section>

<footer><div>weights first · activations for calibration · every claim cross-family</div><div>residual-thoughts.com</div></footer>
</div></body></html>
"""

out = ROOT / "ledger_post.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out} ({len(HTML)/1024:.0f} KB)")
