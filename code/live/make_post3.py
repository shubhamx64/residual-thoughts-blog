"""Generate post #3 for residual-thoughts.com: the polysemanticity-crowding
census. Self-contained single-file HTML, all figures inline SVG from the E5 /
E5b / E5c result JSONs. Output: post3_census.html at repo root.

Visual language (CSS, palette, SVG idiom) matches make_post.py so it drops into
the same blog.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E5 = ROOT / "e5-polysemanticity-census" / "results"
MODELS = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]
MC = {"qwen2.5-1.5b": "#2a78d6", "gemma-2-2b": "#eda100",
      "pythia-1.4b": "#4a3aa7", "tinyllama-1.1b": "#1baf7a"}
MLABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "gemma-2-2b": "Gemma-2-2B",
          "pythia-1.4b": "Pythia-1.4B", "tinyllama-1.1b": "TinyLlama-1.1B"}
MSHORT = {m: MLABEL[m].split("-")[0] for m in MODELS}
INK, MUTED, GRID = "#16202b", "#5c6b7a", "#d7dde2"
CTRL = "#e34948"   # "instrument when the law is real"


def j(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


w = {m: j(E5 / m / "e5_stats.json") for m in MODELS}
r = {m: j(E5 / m / "e5b_stats.json") for m in MODELS}
planted = j(E5 / "toy" / "planted_control.json")
trained = j(E5 / "toy" / "toy_control_h512.json")


def _ticks(lo, hi, n=6):
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10 ** math.floor(math.log10(span / n))
    for mm in (1, 2, 2.5, 5, 10):
        if span / (step * mm) <= n:
            step *= mm
            break
    t0 = math.ceil(lo / step) * step
    out, t = [], t0
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def svg_open(w_, h_):
    return [f'<svg viewBox="0 0 {w_} {h_}" role="img" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:{w_}px;width:100%;height:auto;font-family:inherit">']


# ---- Figure 1: horizontal dot plot with a shaded null band ----
def dotplot_nullband(rows, title, w_=680, h_=None, xlo=-0.2, xhi=0.7,
                     band=0.10, thr=0.15):
    """rows: list of dicts {label, value, color, shape('filled'|'open'|'ref'),
    group_gap(bool)}."""
    ml, mr, mt, mb = 176, 20, 40, 40
    row_h = 26
    h_ = h_ or mt + mb + row_h * len(rows)
    pw = w_ - ml - mr
    ph = h_ - mt - mb

    def X(v):
        return ml + (v - xlo) / (xhi - xlo) * pw
    p = svg_open(w_, h_)
    p.append(f'<text x="{ml}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{INK}">{title}</text>')
    # null band
    p.append(f'<rect x="{X(-band):.1f}" y="{mt}" width="{X(band)-X(-band):.1f}" '
             f'height="{ph:.1f}" fill="#eef1f3"/>')
    p.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{mt}" y2="{mt+ph:.1f}" '
             f'stroke="{MUTED}" stroke-width="1"/>')
    # detection threshold
    p.append(f'<line x1="{X(thr):.1f}" x2="{X(thr):.1f}" y1="{mt}" y2="{mt+ph:.1f}" '
             f'stroke="{INK}" stroke-width="1" stroke-dasharray="4 4"/>')
    p.append(f'<text x="{X(thr):.1f}" y="{mt-6}" font-size="9.5" fill="{INK}" '
             f'text-anchor="middle">detection floor +{thr:g}</text>')
    p.append(f'<text x="{X(0):.1f}" y="{mt+ph+26:.1f}" font-size="10" '
             f'fill="{MUTED}" text-anchor="middle">null band |ρ| &lt; {band:g}</text>')
    # x ticks
    for t in _ticks(xlo, xhi):
        p.append(f'<text x="{X(t):.1f}" y="{mt+ph+14:.1f}" font-size="10" '
                 f'fill="{MUTED}" text-anchor="middle">{t:+g}</text>')
    y = mt + row_h / 2
    for row in rows:
        if row.get("group_gap"):
            p.append(f'<line x1="{ml}" x2="{w_-mr}" y1="{y-row_h/2:.1f}" '
                     f'y2="{y-row_h/2:.1f}" stroke="{GRID}"/>')
        col = row["color"]
        p.append(f'<text x="{ml-12}" y="{y+3.5:.1f}" font-size="11.5" '
                 f'fill="{INK}" text-anchor="end">{row["label"]}</text>')
        x = X(row["value"])
        # stem to zero
        p.append(f'<line x1="{X(0):.1f}" x2="{x:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                 f'stroke="{col}" stroke-width="1.5" opacity="0.5"/>')
        sh = row.get("shape", "filled")
        if sh == "open":
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#fff" '
                     f'stroke="{col}" stroke-width="2"/>')
        elif sh == "ref":
            p.append(f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" '
                     f'height="11" rx="2" fill="{col}"/>')
        else:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}"/>')
        p.append(f'<text x="{x+10:.1f}" y="{y+3.5:.1f}" font-size="10" '
                 f'fill="{MUTED}">{row["value"]:+.2f}</text>')
        y += row_h
    # legend
    lx = ml
    for lab, sh in (("write interface ●", "filled"), ("read interface ○", "open")):
        p.append(f'<text x="{lx}" y="{h_-6}" font-size="10" fill="{MUTED}">{lab}</text>')
        lx += 130
    p.append("</svg>")
    return "".join(p)


# ---- Figure 2: per-layer strip plot ----
def strip_layers(title, w_=680, h_=270, xlo=-0.25, xhi=0.7):
    ml, mr, mt, mb = 120, 20, 40, 40
    pw = w_ - ml - mr
    rows = MODELS
    row_h = (h_ - mt - mb) / len(rows)

    def X(v):
        return ml + (v - xlo) / (xhi - xlo) * pw
    p = svg_open(w_, h_)
    p.append(f'<text x="{ml}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{INK}">{title}</text>')
    p.append(f'<rect x="{X(-0.10):.1f}" y="{mt}" width="{X(0.10)-X(-0.10):.1f}" '
             f'height="{h_-mt-mb:.1f}" fill="#eef1f3"/>')
    p.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{mt}" y2="{h_-mb:.1f}" '
             f'stroke="{MUTED}"/>')
    # planted control reference line
    p.append(f'<line x1="{X(0.52):.1f}" x2="{X(0.52):.1f}" y1="{mt}" y2="{h_-mb:.1f}" '
             f'stroke="{CTRL}" stroke-width="1.5" stroke-dasharray="5 3"/>')
    p.append(f'<text x="{X(0.52):.1f}" y="{mt-6}" font-size="9.5" fill="{CTRL}" '
             f'text-anchor="middle">planted law +0.52</text>')
    for t in _ticks(xlo, xhi):
        p.append(f'<text x="{X(t):.1f}" y="{h_-mb+14:.1f}" font-size="10" '
                 f'fill="{MUTED}" text-anchor="middle">{t:+g}</text>')
    for i, m in enumerate(rows):
        yc = mt + row_h * (i + 0.5)
        col = MC[m]
        p.append(f'<text x="{ml-12}" y="{yc+3.5:.1f}" font-size="11" '
                 f'fill="{INK}" text-anchor="end">{MSHORT[m]}</text>')
        for side, arr in (("w", w[m]["partial_rho_by_layer"]["density"]),
                          ("r", r[m]["partial_rho_by_layer"]["density"])):
            dy = -4 if side == "w" else 4
            for v in arr:
                op = 0.55 if side == "w" else 0.4
                p.append(f'<circle cx="{X(v):.1f}" cy="{yc+dy:.1f}" r="2.6" '
                         f'fill="{col}" opacity="{op}"/>')
        # median markers
        mw = w[m]["partial_rho_median"]["density"]
        p.append(f'<line x1="{X(mw):.1f}" x2="{X(mw):.1f}" y1="{yc-9:.1f}" '
                 f'y2="{yc+9:.1f}" stroke="{col}" stroke-width="2.4"/>')
    p.append(f'<text x="{X(0):.1f}" y="{h_-6}" font-size="10" fill="{MUTED}" '
             f'text-anchor="middle">each dot = one layer · thick tick = median · '
             f'upper row write, lower row read</text>')
    p.append("</svg>")
    return "".join(p)


# ---- Figure 3: strata dumbbell (the sign flip) ----
def dumbbell_strata(title, w_=680, h_=250):
    ml, mr, mt, mb = 120, 90, 40, 34
    pw = w_ - ml - mr
    vals = {m: w[m]["strata"]["rate_matched_median"] for m in MODELS}
    allv = [vals[m][k] for m in MODELS for k in ("isolated", "uncoupled-crowded")]
    xlo, xhi = min(allv) - 0.02, max(allv) + 0.02
    row_h = (h_ - mt - mb) / len(MODELS)

    def X(v):
        return ml + (v - xlo) / (xhi - xlo) * pw
    p = svg_open(w_, h_)
    p.append(f'<text x="{ml}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{INK}">{title}</text>')
    for t in _ticks(xlo, xhi, 5):
        p.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{h_-mb:.1f}" '
                 f'stroke="{GRID}"/>')
        p.append(f'<text x="{X(t):.1f}" y="{h_-mb+14:.1f}" font-size="10" '
                 f'fill="{MUTED}" text-anchor="middle">{t:.2f}</text>')
    for i, m in enumerate(MODELS):
        yc = mt + row_h * (i + 0.5)
        iso, cr = vals[m]["isolated"], vals[m]["uncoupled-crowded"]
        col = MC[m]
        p.append(f'<text x="{ml-12}" y="{yc+3.5:.1f}" font-size="11" '
                 f'fill="{INK}" text-anchor="end">{MSHORT[m]}</text>')
        p.append(f'<line x1="{X(iso):.1f}" x2="{X(cr):.1f}" y1="{yc:.1f}" '
                 f'y2="{yc:.1f}" stroke="{col}" stroke-width="2"/>')
        p.append(f'<circle cx="{X(iso):.1f}" cy="{yc:.1f}" r="5.5" fill="#fff" '
                 f'stroke="{col}" stroke-width="2"/>')
        p.append(f'<circle cx="{X(cr):.1f}" cy="{yc:.1f}" r="5.5" fill="{col}"/>')
        lean = "crowded more mixed →" if cr > iso else "← isolated more mixed"
        p.append(f'<text x="{w_-mr+6}" y="{yc+3.5:.1f}" font-size="9.5" '
                 f'fill="{MUTED}">{lean}</text>')
    p.append(f'<text x="{ml}" y="{h_-6}" font-size="10" fill="{MUTED}">'
             f'○ isolated neurons   ● crowded neurons · rate-matched median entropy (bits)</text>')
    p.append("</svg>")
    return "".join(p)


# ---- Figure 4: read-write correlation bars ----
def hbar_readwrite(title, w_=680, h_=210):
    ml, mr, mt, mb = 120, 60, 40, 30
    pw = w_ - ml - mr
    row_h = (h_ - mt - mb) / len(MODELS)
    xhi = 1.0

    def X(v):
        return ml + v / xhi * pw
    p = svg_open(w_, h_)
    p.append(f'<text x="{ml}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{INK}">{title}</text>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{h_-mb:.1f}" '
                 f'stroke="{GRID}"/>')
        p.append(f'<text x="{X(t):.1f}" y="{h_-mb+14:.1f}" font-size="10" '
                 f'fill="{MUTED}" text-anchor="middle">{t:g}</text>')
    p.append(f'<line x1="{X(1.0):.1f}" x2="{X(1.0):.1f}" y1="{mt}" y2="{h_-mb:.1f}" '
             f'stroke="{INK}" stroke-dasharray="4 4"/>')
    p.append(f'<text x="{X(1.0):.1f}" y="{mt-6}" font-size="9" fill="{INK}" '
             f'text-anchor="end">“one place” = 1.0</text>')
    for i, m in enumerate(MODELS):
        yc = mt + row_h * (i + 0.5)
        v = r[m]["read_write_density_rho_median"]
        col = MC[m]
        p.append(f'<text x="{ml-12}" y="{yc+3.5:.1f}" font-size="11" '
                 f'fill="{INK}" text-anchor="end">{MSHORT[m]}</text>')
        p.append(f'<rect x="{ml}" y="{yc-7:.1f}" width="{X(v)-ml:.1f}" height="14" '
                 f'rx="3" fill="{col}"/>')
        p.append(f'<text x="{X(v)+6:.1f}" y="{yc+3.5:.1f}" font-size="10.5" '
                 f'fill="{INK}">{v:.2f}</text>')
    p.append("</svg>")
    return "".join(p)


fig1 = dotplot_nullband([
    {"label": "Qwen · write", "value": w["qwen2.5-1.5b"]["partial_rho_median"]["density"], "color": MC["qwen2.5-1.5b"]},
    {"label": "Qwen · read", "value": r["qwen2.5-1.5b"]["partial_rho_median"]["density"], "color": MC["qwen2.5-1.5b"], "shape": "open"},
    {"label": "Gemma-2 · write", "value": w["gemma-2-2b"]["partial_rho_median"]["density"], "color": MC["gemma-2-2b"]},
    {"label": "Gemma-2 · read", "value": r["gemma-2-2b"]["partial_rho_median"]["density"], "color": MC["gemma-2-2b"], "shape": "open"},
    {"label": "Pythia · write", "value": w["pythia-1.4b"]["partial_rho_median"]["density"], "color": MC["pythia-1.4b"]},
    {"label": "Pythia · read", "value": r["pythia-1.4b"]["partial_rho_median"]["density"], "color": MC["pythia-1.4b"], "shape": "open"},
    {"label": "TinyLlama · write", "value": w["tinyllama-1.1b"]["partial_rho_median"]["density"], "color": MC["tinyllama-1.1b"]},
    {"label": "TinyLlama · read", "value": r["tinyllama-1.1b"]["partial_rho_median"]["density"], "color": MC["tinyllama-1.1b"], "shape": "open"},
    {"label": "planted law (control)", "value": planted["partial_rho_density"], "color": CTRL, "shape": "ref", "group_gap": True},
    {"label": "same control, shuffled", "value": planted["shuffle_rho_median_abs"], "color": MUTED, "shape": "ref"},
], "Partial ρ(regime mixing, crowding | firing rate) – eight real measurements, one positive control")

fig2 = strip_layers("Every layer, not just the median – write and read density, four families")
fig3 = dumbbell_strata("The only above-noise effects flip direction between families")
fig4 = hbar_readwrite("Is a neuron “in a crowded place”? Read-vs-write crowding agreement per neuron")

# numbers for prose
def med(m, side, k):
    d = (w if side == "w" else r)[m]
    return d["partial_rho_median"][k]


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polysemanticity Is Not the Price of Packing – a four-family, two-interface, instrumented null</title>
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --line:#c6cfd6; --paper:#edf0f2; --card:#f8fafb;
  --blue:#2a78d6; --blue-soft:#dde7fa; --amber:#c46a10; --red:#e34948; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink);
  font-family:Georgia,'Times New Roman',serif; font-size:17.5px; line-height:1.65; }}
.wrap {{ max-width:960px; margin:0 auto; padding:0 22px 80px; }}
.measure {{ max-width:700px; }}
header {{ border-bottom:1px solid var(--line); padding:18px 0; margin-bottom:44px;
  font-family:ui-monospace,Consolas,monospace; font-size:13px; color:var(--muted);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
header b {{ color:var(--ink); }}
h1 {{ font-family:system-ui,'Segoe UI',sans-serif; font-weight:800;
  font-size:clamp(30px,5.5vw,50px); line-height:1.05; letter-spacing:-0.02em; max-width:860px; }}
.dek {{ font-size:20px; color:var(--muted); font-style:italic; max-width:700px; margin:18px 0 8px; }}
.dek b {{ color:var(--ink); font-style:normal; }}
.status {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden; margin:34px 0 8px; }}
.status div {{ background:var(--card); padding:13px 15px; }}
.status .k {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); display:block; }}
.status .v {{ font-family:system-ui,sans-serif; font-weight:700; font-size:15px; }}
.status .pass {{ color:#0a7a3d; }} .status .null {{ color:var(--amber); }}
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
.ctrlbox {{ border-left-color:var(--red); }} .ctrlbox .tag {{ color:var(--red); }}
table {{ border-collapse:collapse; margin:22px 0; font-size:14.5px; width:100%;
  font-family:system-ui,'Segoe UI',sans-serif; }}
th {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); text-align:left; padding:8px 10px;
  border-bottom:2px solid var(--ink); }}
td {{ padding:9px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
tr.hl td {{ background:var(--blue-soft); font-weight:650; }}
.tblwrap {{ overflow-x:auto; }}
figure {{ margin:26px 0; background:var(--card); border:1px solid var(--line);
  border-radius:6px; padding:18px 16px 10px; }}
figcaption {{ font-size:13.5px; color:var(--muted); padding:8px 6px 4px; line-height:1.5; }}
.note {{ font-size:14.5px; color:var(--muted); }}
.pull {{ border-left:3px solid var(--blue); padding:4px 0 4px 20px; margin:26px 0;
  font-size:19px; font-style:italic; max-width:660px; }}
.callout {{ background:#fbf6ec; border:1px solid #e6d8b8; border-radius:6px;
  padding:16px 20px; margin:24px 0; max-width:720px; }}
.callout h3 {{ margin-top:0; color:var(--amber); }}
footer {{ margin-top:70px; border-top:1px solid var(--line); padding-top:22px;
  font-family:ui-monospace,Consolas,monospace; font-size:12.5px; color:var(--muted);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
code {{ font-family:ui-monospace,Consolas,monospace; font-size:.85em; background:var(--card);
  border:1px solid var(--line); border-radius:3px; padding:1px 5px; }}
a {{ color:var(--blue); }}
</style></head><body><div class="wrap">

<header><div><b>residual thoughts</b> / working notes · no. 3</div>
<div>July 5, 2026 · four model families · one RTX 4060 Ti</div></header>

<h1>Polysemanticity Is Not the Price of Packing<br>
<span style="color:var(--muted);font-weight:700">at least not at the neuron level – a four-family, two-interface, instrumented null</span></h1>

<p class="dek">Superposition theory has a folk corollary: the neurons that <b>mix meanings</b> should be the ones whose weight vectors <b>crowd together</b>, because crowding is what packing many features into few dimensions looks like. It is a clean, testable claim. We tested it – dictionary-free, pre-registered, on both faces of the MLP, across four model families – and it is false at neuron granularity. Then we built a positive control to prove the test could have seen the law if it were there.</p>

<div class="status">
<div><span class="k">write interface</span><span class="v null">NULL · 4/4</span></div>
<div><span class="k">read interface</span><span class="v null">NULL · 4/4</span></div>
<div><span class="k">positive control</span><span class="v pass">ρ +0.52</span></div>
<div><span class="k">proxy validity</span><span class="v pass">ρ +0.83</span></div>
</div>

<!-- ===== THE CLAIM ===== -->
<section>
<div class="eyebrow">The claim under test</div>
<h2>Crowding and mixing, made measurable without a dictionary</h2>
<p class="measure">In superposition, a network represents more features than it has dimensions by letting feature directions overlap. The intuitive reading at the <em>neuron</em> level: a neuron sitting in a geometrically crowded neighborhood is being asked to serve many masters, so it should fire across many unrelated contexts – it should be <strong>polysemantic</strong>. If that held, weight geometry alone would hand you a zero-data polysemanticity map: find the crowded neurons, you have found the mixed ones.</p>
<p class="measure">We make both sides concrete without ever touching an SAE:</p>
<div class="tblwrap"><table>
<tr><th>quantity</th><th>definition</th><th>basis</th></tr>
<tr><td>mixing</td><td>Shannon entropy of a neuron's firing distribution over 5 task regimes (math, math-prose, code, code-prose, prose), rates normalized by per-class token counts</td><td>activation, ~4-min forward capture</td></tr>
<tr><td>crowding</td><td>neighborhood density = number of same-layer neurons with |cos| &gt; 0.4 to this one (plus max|cos|, top-10 mean as secondaries)</td><td>weights only</td></tr>
</table></div>
<p class="measure">Every correlation below is a <strong>partial</strong> Spearman ρ, controlling for log firing rate – a busy neuron is mechanically both more mixed and more likely to look crowded, and that confound is removed by rank-residualizing on rate before correlating. Neurons with &lt; 50 firing events are excluded. Seeds fixed; four families spanning gated (Qwen, Gemma-2, TinyLlama) and plain (Pythia) MLPs, chosen so a Gemma-only quirk cannot masquerade as a law.</p>
</section>

<!-- ===== THE HEADLINE ===== -->
<section>
<div class="eyebrow">The result</div>
<h2>Everything hugs zero – on both faces of the neuron</h2>
<div class="verdict neg"><span class="tag">Verdict – NULL, 4/4 families, both interfaces</span><br>
The median-across-layers partial ρ between mixing and crowding is <strong>+0.00 to +0.05 on the write interface</strong> (down-projection columns) and <strong>−0.005 to +0.033 on the read interface</strong> (gate-projection rows) – every one inside the pre-registered |ρ| &lt; 0.10 null band. No couple-type stratum separates by more than 0.12 bits at matched firing rate. Robust across firing thresholds (q98/q99.5), a 4-class entropy variant, and an outside-top-class proxy; the null <em>strengthens</em> when universal-neuron candidates are removed.</div>
<figure>{fig1}<figcaption>Each point is the median-across-layers partial correlation for one model on one interface. Filled = write, open = read. The gray band is the pre-registered null; the dashed line is the +0.15 threshold a real effect had to clear. Eight of eight real measurements sit in the band. The red square is the <b>same statistic run on synthetic data with a crowding→mixing law planted by construction</b> – it lands at +0.52. The instrument is not blind; there is simply nothing to see in the real models.</figcaption></figure>
<p class="measure">The write side was the natural place to look – a neuron's output direction is what collides in the residual stream. But recruitment is decided on the <em>read</em> side: whether a neuron fires is a property of its input weights, not its output. So the read interface was the last place a neuron-level packing law could hide. It doesn't hide there either.</p>
<figure>{fig2}<figcaption>The medians are not hiding a wide spread: every individual layer's partial ρ, all four families, both interfaces. The cloud sits on zero at every depth. The red dashed line marks where the planted-law control lands – the entire real distribution is an order of magnitude short of it.</figcaption></figure>
</section>

<!-- ===== WHY (decorrelation) ===== -->
<section>
<div class="eyebrow">Why the intuition fails</div>
<h2>A neuron is not “in a crowded place.” It is crowded per interface.</h2>
<p class="measure">The read-side test was not a redundant re-run of the write-side test, and the reason is a finding in itself. If a neuron simply occupied a crowded or an uncrowded region of the model, its read-crowding and write-crowding would move together – the correlation would be near 1. It is <strong>ρ 0.21 to 0.47</strong>.</p>
<figure>{fig4}<figcaption>Per-neuron agreement between read-side and write-side neighborhood density, per family. Recruitment geometry and expression geometry are substantially independent properties of the same unit. There is no single scalar “this neuron is crowded” – so there is no single crowding for mixing to track in the first place.</figcaption></figure>
<p class="measure">That decorrelation quietly explains the whole null. The folk picture assumes one geometric fact per neuron. There are at least two, and they disagree, and neither one predicts what the neuron actually does across regimes.</p>
</section>

<!-- ===== THE SIGN FLIP ===== -->
<section>
<div class="eyebrow">The tell</div>
<h2>The few effects that clear the noise point in opposite directions</h2>
<p class="measure">A weak-but-real law would at least be <em>consistent</em>: crowded neurons a little more mixed in every family. Instead the largest excursions cancel. On the write side the only two measures that graze the band – Gemma-2's local-crowding +0.12 and Pythia's top-10 −0.12 – have opposite signs, one family each, two hundredths past a threshold chosen as the noise floor. The stratum medians tell the same story in a picture.</p>
<figure>{fig3}<figcaption>Rate-matched median mixing for a family's most-isolated vs most-crowded neurons. A universal crowding→mixing law would lean every bar the same way. Gemma-2's isolated neurons are its <b>cleanest</b>; TinyLlama's isolated neurons are its <b>most mixed</b> – the effect reverses between architectures. Sign-flips across families and measures are the signature of noise around zero, and they, not any single point, carry the argument.</figcaption></figure>
<p class="measure">Gemma-2 showing the lone sub-threshold positive is exactly the pattern to expect: across this whole research program Gemma-2 is repeatedly the most weight-legible model, so a whiff of structure→function signal there, and nowhere else, reads as a property of Gemma, not of transformers. It was pre-registered as not counting.</p>
</section>

<!-- ===== POSITIVE CONTROL ===== -->
<section>
<div class="eyebrow">The insurance</div>
<h2>A null is only worth as much as the instrument behind it</h2>
<p class="measure">The obvious attack on any negative result: maybe the pipeline couldn't detect the effect even if it existed – maybe five coarse regimes can't resolve real polysemanticity. That objection is answerable in-house, and cheaply, with two toy systems.</p>

<div class="verdict ctrlbox"><span class="tag">Positive control – PASS, both flanks</span><br>
<strong>Detection:</strong> on synthetic data with a graded crowding→mixing law planted by construction, the <em>identical</em> statistic recovers it at partial ρ <strong>+0.52</strong> (density) / +0.63 (top-10), while the same test on label-shuffled data stays at q95 |ρ| 0.094 – inside the null band. <strong>Proxy validity:</strong> in a trained Toy-Models-style autoencoder where each neuron's true feature composition is known, the coarse 5-regime entropy tracks ground-truth mixing at ρ <strong>+0.83</strong>. The measure sees mixing; the pipeline sees the law when the law is there.</div>

<div class="tblwrap"><table>
<tr><th>quantity</th><th>value</th><th>bar</th><th>reading</th></tr>
<tr class="hl"><td>real census, write (best family)</td><td>+0.05</td><td>–</td><td>the null</td></tr>
<tr><td>planted law, density</td><td>+{planted['partial_rho_density']:.2f}</td><td>≥ +0.15</td><td>detected, 3–4× over floor</td></tr>
<tr><td>planted law, top-10</td><td>+{planted['partial_rho_top10']:.2f}</td><td>agrees in sign</td><td>detected</td></tr>
<tr><td>planted law, labels shuffled (q95)</td><td>{planted['shuffle_rho_q95_abs']:.3f}</td><td>&lt; 0.10</td><td>no false positive</td></tr>
<tr><td>census entropy vs ground-truth mixing</td><td>+{trained['ground_truth']['instrument_rho_entropy']:.2f}</td><td>–</td><td>proxy is faithful</td></tr>
</table></div>

<div class="callout">
<h3>An artifact worth naming</h3>
<p class="measure" style="margin-bottom:0">The first planted control used a clean two-group split (isolated vs crowded), which made the crowding variable perfectly bimodal. Pushed through the rank-residualization, a near-binary regressor inflated the permutation null to q95 ≈ 0.16 – which would have made the instrument look <em>less</em> sensitive than it is. The fix was to regrade the planted data to a continuous crowding level matching the real census's density spread; the null fell back to ≈0.04 and the detection signal was unchanged (bimodal +0.83, graded +0.52). <strong>The regrade fixed the null estimate, not the signal</strong> – the kind of thing a control has to get right, and the kind of thing that is only visible if you keep both runs. Both are in the repo.</p>
</div>

<div class="callout">
<h3>The mechanistic “why,” seen in a toy</h3>
<p class="measure" style="margin-bottom:0">The trained autoencoder carries feature-level superposition by construction – 2560 sparse features in 512 neurons, 5× compression, dense packing, clean reconstruction. Yet its neuron write-columns barely crowd at all: maximum pairwise |cos| across the whole layer is <strong>{trained['ground_truth']['max_pair_cos_write']:.2f}</strong>. Here is a minimal system where superposition is present and heavy, and it still does not imprint as neuron-level geometric crowding – because features and neurons are simply different objects. That is the decoupling thesis shown where you can see both levels at once.</p>
</div>
</section>

<!-- ===== SCOPE ===== -->
<section>
<div class="eyebrow">What this does and doesn't say</div>
<h2>Scope, stated plainly</h2>
<p class="measure">This is a claim about <strong>neurons</strong>, not features. Superposition theory is fundamentally about features as linear combinations of neurons; an SAE feature can be crowded and polysemantic in ways no single neuron reveals. Nothing here refutes feature-level superposition. What it refutes is the <em>neuron-granularity shortcut</em> – the hope that you could read polysemanticity off raw weight geometry without a dictionary. You cannot, on either interface, in any of four families.</p>
<p class="measure">It is also a claim about <strong>task-regime</strong> mixing (five coarse classes), not fine-grained semantic mixing, and about <strong>MLP</strong> geometry, not attention. Within those bounds it is as airtight as we could make it: pre-registered thresholds, both faces of the neuron, four architectures, a positive control at 3–4× the detection floor, and a demonstrated-faithful mixing proxy.</p>
<div class="pull">Which neurons mix regimes is set by the statistics of the data a model was trained on, not by the geometric necessity of packing. The ledger of a model's capacity has to be <em>read from usage</em>; it is not sitting latent in the Gram matrix of its weights.</div>
<p class="measure note">This is the negative half of a longer weights-first program (earlier notes: a routing map of attention heads, and its errata). Where those notes showed what weight geometry <em>does</em> know – gain topography, protection maps worth two-thirds of a Fisher signal at zero data cost – this one marks a firm boundary on what it does not. Pre-registrations, per-model JSONs, and the toy controls are in the repo, each experiment with a verdict-first report. Four families, two interfaces, two toys, one consumer GPU.</p>
</section>

<footer><div>weights first · activations for calibration · every claim cross-family</div>
<div>residual-thoughts.com · no. 3</div></footer>
</div></body></html>
"""

out = ROOT / "post3_census.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out} ({len(HTML)/1024:.0f} KB)")
