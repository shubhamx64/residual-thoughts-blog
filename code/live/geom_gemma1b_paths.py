# geom_gemma1b_paths_plus.py
import os, math, json, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
import umap
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend to avoid Tk in threads
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-pt"
OUTDIR = Path("outputs"); OUTDIR.mkdir(exist_ok=True, parents=True)
SAVE_PLOTS = True
MAX_NEW = 256
USE_WHITEN = False             # toggle per-layer whitening for metrics
SPS_BINS = 32                  # spectral bins for path signature
UMAP_NEIGH = 25
UMAP_MINDIST = 0.05
PCA_POINTS_STRIDE = 1          # plot every kth token to declutter
SEGMENT_LEN_FOR_LINES = 6      # draw polyline in short segments (reduces UMAP starbursts); set 0 to disable lines
SEED = 42

# Token-index windows inside the continuation for windowed metrics.
# "None" means "to the end".
TOK_WINDOWS = [
    ("early", 0, 32),
    ("mid",   32, 64),
    ("late",  -64, None),      # last 64 tokens (if available)
]

# Optional fixed bands for "corridor" summary; leave as None to discover by trough.
BANDS = dict(
    early=range(0, 5),
    corridor=range(5, 14),
    late=range(14, 26),
    final=range(26, 27),
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------
# 0) load model + tokenizer
# -----------------------------
print("Loading model:", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")
model.eval()
config = AutoConfig.from_pretrained(MODEL_ID)
text_cfg = getattr(config, "text_config", config)  # gemma-3 packs text settings under text_config
print("n_layers:", text_cfg.num_hidden_layers)
print("hidden_size:", text_cfg.hidden_size)
print("n_heads:", text_cfg.num_attention_heads)
print("head_dim:", text_cfg.hidden_size // text_cfg.num_attention_heads)

def tree(mod, prefix="", depth=0, max_depth=1):
    if depth > max_depth: return
    for name, child in mod.named_children():
        print(prefix + name, "->", child.__class__.__name__)
        tree(child, prefix + "  ", depth+1, max_depth)
print("\n=== Module tree (top few levels) ===")
tree(model, max_depth=1)

# -----------------------------
# 1) PathBank (diversified, reproducible)
# -----------------------------
BASES_POOL = [
    # dynamics / emergence
    "Why do large cities keep growing even when they already seem overcrowded?",
    "How do small local choices add up to traffic jams on a highway?",
    #"Why do some technologies suddenly take off after years of seeming niche?",
    #"What makes certain online communities stay healthy while others decay over time?",

    # coordination / institutions
    "What typically goes wrong in large cross-functional projects, and how can it be reduced?",
    "How should a country organize decision-making during a fast-moving crisis?",
    #"When is it better to rely on markets versus central planning for allocating scarce resources?",
    #"What mechanisms actually improve trust between citizens and institutions?",
]

PERS = {
    "cs":      "Answer from a computer science perspective.",
    "econ":    "Answer from an economic systems perspective.",
    "policy":  "Answer from a public-policy and governance perspective.",
    "neuro":   "Answer from a neuroscience and cognitive-science perspective.",
}

MAX_BASES = 24  # subsample for runtime; adjust as needed

def prompts():
    rng = np.random.default_rng(SEED)
    pool = BASES_POOL.copy(); rng.shuffle(pool)
    chosen = pool[:min(MAX_BASES, len(pool))]
    rows = []
    for i, base in enumerate(chosen):
        for c, tail in PERS.items():
            text = f"{base}\n{tail}"
            rows.append({"id": f"q{i}_{c}", "perspective": c, "text": text})
    return rows
ROWS = prompts()

# -----------------------------
# 2) helpers
# -----------------------------
@torch.inference_mode()
def generate_then_capture(text, max_new=MAX_NEW):
    inpt = tok(text, return_tensors="pt").to(model.device)
    prompt_len = inpt["input_ids"].shape[1]
    gen = model.generate(**inpt, do_sample=False, max_new_tokens=max_new)
    full_ids = gen[0]
    out = model(input_ids=full_ids.unsqueeze(0), output_hidden_states=True, return_dict=True)
    Hs = [h[0].to("cpu") for h in out.hidden_states]  # list of (seq, hidden)
    return full_ids.cpu(), int(prompt_len), Hs

def l2_normalize_rows(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def whiten_tokens(H):
    Hc = H - H.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Hc, full_matrices=False)
    return (Hc @ Vt.T) / (S + 1e-6)

def step_vecs(H):           # (T,D) -> (T-1,D)
    return H[1:] - H[:-1]

def step_dists(H):          # (T,D) -> (T-1,)
    return np.linalg.norm(step_vecs(H), axis=1)

def curvature(H):           # (T,D) -> (T-2,)
    v = step_vecs(H)
    a = v[1:] - v[:-1]
    return np.linalg.norm(a, axis=1)

def spectral_signature(Hhat, bins=SPS_BINS):
    P = PCA(n_components=min(3, Hhat.shape[1]), random_state=SEED).fit_transform(Hhat)
    F = np.abs(np.fft.rfft(P, axis=0)).mean(axis=1)  # (T_fft,)
    K = min(bins, len(F))
    return F[:K]

def spectral_stats(Hhat):
    """Return (lowfreq_energy_ratio, centroid) from rFFT over token axis."""
    P = PCA(n_components=min(3, Hhat.shape[1]), random_state=SEED).fit_transform(Hhat)
    F = np.abs(np.fft.rfft(P, axis=0)).mean(axis=1)  # (M,)
    if F.sum() == 0: 
        return 0.0, 0.0
    m = len(F)
    half = max(1, m//6)            # define 'low' as first ~1/6 of spectrum
    low_ratio = F[:half].sum() / F.sum()
    centroid = (np.arange(m) * F).sum() / F.sum()
    return float(low_ratio), float(centroid)

def window_slice(H, start, end):
    """Slice continuation matrix H (T,D) by token indices; supports negative start."""
    T = H.shape[0]
    s = start if start is not None else 0
    e = end if end is not None else T
    if s < 0: s = max(0, T + s)
    e = min(T, e)
    if s >= e: return H[:0]
    return H[s:e]

def rec_base(rec_id: str):
    return rec_id.split("_")[0]

# -----------------------------
# Precompute features and LOBO scorer helpers
# -----------------------------
def get_lab_map(records):
    """Build a label map from all perspectives present in the records."""
    uniq = sorted({rec["perspective"] for rec in records})
    return {p: i for i, p in enumerate(uniq)}
def _vectorize_pf(pf, part: str):
    if part == "geom":
        return pf["geom"]
    elif part == "mom":
        return pf["mom"]
    elif part == "spec":
        return pf["spec"]
    else:
        return np.concatenate([pf["geom"], pf["mom"], pf["spec"]])

def build_features(records, use_whiten=False, part="all"):
    lab_map = get_lab_map(records)
    per_layer = []
    for li in range(L):
        feats, y, bases = [], [], []
        for rec in records:
            Hc = cont_layer(rec, li)
            pf = path_features(Hc, use_whiten=use_whiten)
            if pf is None:
                continue
            vec = _vectorize_pf(pf, part)
            feats.append(vec)
            y.append(lab_map[rec["perspective"]])
            bases.append(rec_base(rec["id"]))
        if feats:
            per_layer.append((np.stack(feats), np.array(y), np.array(bases)))
        else:
            per_layer.append(None)
    return per_layer

def lobo_score_for_layer(X, y, bases):
    uniq = sorted(set(bases.tolist()))
    correct, total = 0, 0
    for b in uniq:
        train = bases != b
        test = bases == b
        if test.sum() == 0 or train.sum() < 6:
            continue
        clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                            RidgeClassifierCV(alphas=np.logspace(-3,3,9)))
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        correct += (pred == y[test]).sum()
        total += test.sum()
    return (correct/total) if total>0 else np.nan

# -----------------------------
# 3) capture all paths (ALL layers)
# -----------------------------
all_records = []
print("\nCapturing hidden states...")
for row in tqdm(ROWS):
    ids, prompt_len, Hs = generate_then_capture(row["text"])
    rec = dict(id=row["id"], perspective=row["perspective"], text=row["text"],
               ids=ids, prompt_len=prompt_len,
               layers=[h.to(torch.float32).numpy() for h in Hs])
    all_records.append(rec)

print("\nHidden-state shapes for first example across layers:")
for li, H in enumerate(all_records[0]["layers"]):
    print(f"L{li}: {H.shape}")

L = len(all_records[0]["layers"])

def cont_layer(rec, li):
    H = rec["layers"][li]
    p = rec["prompt_len"]
    return H[p:]  # continuation only

# -----------------------------
# 4) geometry stats per layer (global + windowed)
# -----------------------------
metrics_by_layer = []
print("\n=== Cosine-normalized geometry (continuations; mean +/- std) ===")
for li in range(L):
    ds, cs = [], []
    win_stats = {name: {"dlt": [], "curv": []} for name,_,_ in TOK_WINDOWS}
    lowE, cent = [], []
    for rec in all_records:
        Hc = cont_layer(rec, li)
        if Hc.shape[0] < 4: continue
        X = whiten_tokens(Hc) if USE_WHITEN else Hc
        Hhat = l2_normalize_rows(X)

        # global
        ds.append(step_dists(Hhat).mean())
        cs.append(curvature(Hhat).mean())

        # spectrum
        r, c = spectral_stats(Hhat)
        lowE.append(r); cent.append(c)

        # windows
        for name, s, e in TOK_WINDOWS:
            Hw = window_slice(Hhat, s, e)
            if Hw.shape[0] >= 4:
                win_stats[name]["dlt"].append(step_dists(Hw).mean())
                win_stats[name]["curv"].append(curvature(Hw).mean())

    if ds and cs:
        row = {
            "layer": li,
            "dlt_mean": float(np.mean(ds)), "dlt_std": float(np.std(ds)),
            "curv_mean": float(np.mean(cs)), "curv_std": float(np.std(cs)),
            "lowfreq_ratio": float(np.mean(lowE)),
            "spec_centroid": float(np.mean(cent)),
        }
        for name in win_stats:
            if win_stats[name]["dlt"]:
                row[f"{name}_dlt_mean"]  = float(np.mean(win_stats[name]["dlt"]))
                row[f"{name}_dlt_std"]   = float(np.std(win_stats[name]["dlt"]))
                row[f"{name}_curv_mean"] = float(np.mean(win_stats[name]["curv"]))
                row[f"{name}_curv_std"]  = float(np.std(win_stats[name]["curv"]))
        metrics_by_layer.append(row)
        print(f"L{li:02d}: dlt={row['dlt_mean']:.4f} +/- {row['dlt_std']:.4f} | "
              f"curv={row['curv_mean']:.4f} +/- {row['curv_std']:.4f} | "
              f"lowE={row['lowfreq_ratio']:.3f} | cent={row['spec_centroid']:.2f}")

# Save CSV/JSON
import csv, json as _json
csv_path = OUTDIR / "geometry_by_layer.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in metrics_by_layer for k in r.keys()}))
    w.writeheader(); w.writerows(metrics_by_layer)
with open(OUTDIR / "geometry_by_layer.json", "w") as f:
    _json.dump(metrics_by_layer, f, indent=2)

# -----------------------------
# 5) Tangent alignment: within vs across perspectives
# -----------------------------
def tangent_alignment(records):
    """
    For each layer, compute mean cosine between *tangent sequences* v_t across samples.
    Report within-perspective and across-perspective means.
    """
    layers_v = [[] for _ in range(L)]  # each item: list of (perspective, V) with V shape (T-1, D)
    for rec in records:
        for li in range(L):
            Hc = cont_layer(rec, li)
            if Hc.shape[0] < 4: continue
            X = whiten_tokens(Hc) if USE_WHITEN else Hc
            Hhat = l2_normalize_rows(X)
            V = step_vecs(Hhat)
            layers_v[li].append((rec["perspective"], V))

    out=[]
    for li in range(L):
        items = layers_v[li]
        if len(items) < 3: 
            out.append(dict(layer=li, within=np.nan, across=np.nan, delta=np.nan))
            continue
        # align by min length
        m = min(v.shape[0] for _,v in items)
        if m < 2:
            out.append(dict(layer=li, within=np.nan, across=np.nan, delta=np.nan))
            continue

        def flat(v):
            Z = v[:m]
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True)+1e-8)
            return Z.flatten()
        W, A, nW, nA = 0.0, 0.0, 0, 0
        for i in range(len(items)):
            pi, Vi = items[i]
            fi = flat(Vi)
            for j in range(i+1, len(items)):
                pj, Vj = items[j]
                fj = flat(Vj)
                cos = float(np.dot(fi, fj) / (np.linalg.norm(fi)*np.linalg.norm(fj)+1e-8))
                if pi==pj: W += cos; nW += 1
                else:      A += cos; nA += 1
        within = W/nW if nW else np.nan
        across = A/nA if nA else np.nan
        out.append(dict(layer=li, within=within, across=across, delta=(within-across)))
    return out

align = tangent_alignment(all_records)
with open(OUTDIR / "tangent_alignment.json", "w") as f:
    _json.dump(align, f, indent=2)

# Corridor index (low step & curvature, high lowfreq energy) and Divergence index (within-across)
def zscore(x):
    x = np.array(x); return (x - x.mean()) / (x.std()+1e-8)

def compute_indices(geom_rows, align_rows):
    # match by layer
    dlt = np.array([r["dlt_mean"] for r in geom_rows])
    cur = np.array([r["curv_mean"] for r in geom_rows])
    low = np.array([r["lowfreq_ratio"] for r in geom_rows])
    div = np.array([r["delta"] for r in align_rows])

    # corridor: small dlt, small curv, large lowfreq
    CI = -zscore(dlt) + -zscore(cur) + zscore(low)
    # divergence: within - across tangent similarity
    DI = zscore(div)

    rows=[]
    for i in range(len(geom_rows)):
        rows.append(dict(layer=i, corridor_index=float(CI[i]), divergence_index=float(DI[i])))
    return rows

indices = compute_indices(metrics_by_layer, align)
with open(OUTDIR / "indices.json", "w") as f:
    _json.dump(indices, f, indent=2)

# Auto-detect trough for report
dlt_arr = np.array([r["dlt_mean"] for r in metrics_by_layer])
cur_arr = np.array([r["curv_mean"] for r in metrics_by_layer])
corridor_layers = np.argsort(dlt_arr + cur_arr)[:max(1, L//4)]
print("\nHeuristic corridor (lowest dlt+curv):", sorted(corridor_layers.tolist()))

# -----------------------------
# 6) PCA & UMAP plots per layer (windowed segments to reduce starbursts)
# -----------------------------
def path_plot(layer_idx, kind="umap"):
    X = []
    spans = []  # (start,end,perspective)
    lengths=[]
    for rec in all_records:
        Hc = cont_layer(rec, layer_idx)
        if Hc.shape[0] == 0: continue
        Hhat = l2_normalize_rows(Hc)
        s = len(X); X.extend(Hhat[::PCA_POINTS_STRIDE]); e = len(X)
        spans.append((s, e, rec["perspective"]))
        lengths.append(e-s)
    X = np.array(X)
    if len(X) < 10: return

    if kind == "umap":
        reducer = umap.UMAP(n_neighbors=UMAP_NEIGH, min_dist=UMAP_MINDIST, metric="cosine", random_state=SEED)
        Y = reducer.fit_transform(X)
        title = f"UMAP continuation paths @ L{layer_idx}"
        fname = OUTDIR / f"umap_L{layer_idx:02d}.png"
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        Y = reducer.fit_transform(X)
        title = f"PCA continuation paths @ L{layer_idx}"
        fname = OUTDIR / f"pca_L{layer_idx:02d}.png"

    cm = {
        "cs": "tab:blue",
        "econ": "tab:green",
        "policy": "black",
        "neuro": "tab:red",
        # legacy/fallback labels
        "history": "tab:red",
    }
    plt.figure(figsize=(6,5))
    seen=set()
    for (s,e,pers) in spans:
        P = Y[s:e]
        color = cm.get(pers, "gray")
        if SEGMENT_LEN_FOR_LINES and len(P) > SEGMENT_LEN_FOR_LINES:
            # draw short segments
            for k in range(0, len(P)-SEGMENT_LEN_FOR_LINES, SEGMENT_LEN_FOR_LINES):
                Q = P[k:k+SEGMENT_LEN_FOR_LINES+1]
                plt.plot(Q[:,0], Q[:,1], color=color, alpha=0.7, linewidth=1.0)
        else:
            plt.plot(P[:,0], P[:,1], color=color, alpha=0.9, linewidth=1.3)
        plt.scatter(P[0,0], P[0,1], color=color, s=12)
        plt.scatter(P[-1,0], P[-1,1], color=color, s=16, marker="x")
        seen.add(pers)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=cm.get(p, "gray"), lw=2, label=p) for p in sorted(seen)]
    plt.legend(handles=handles, title="Perspective", loc="best")
    plt.title(title); plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(fname, dpi=140); plt.close()
    else:
        plt.show()

print("\nBuilding PCA & UMAP plots per layer...")
for li in range(L):
    path_plot(li, "pca")
    path_plot(li, "umap")
print(f"Saved plots to: {OUTDIR.resolve()}")

# -----------------------------
# 7) LOBO probe with ablations
# -----------------------------
def path_features(H, use_whiten=False):
    X = whiten_tokens(H) if use_whiten else H
    Hhat = l2_normalize_rows(X)
    if Hhat.shape[0] < 4:
        return None
    d = step_dists(Hhat); dd = curvature(Hhat)
    Lp = d.sum(); mstep=d.mean(); mcurv=dd.mean(); scurv=dd.std()
    feat_geom = np.array([Lp, mstep, mcurv, scurv])
    feat_mom  = np.concatenate([Hhat.mean(axis=0), Hhat.std(axis=0)])
    feat_spec = spectral_signature(Hhat, bins=SPS_BINS)
    return dict(geom=feat_geom, mom=feat_mom, spec=feat_spec)

def lobo_ablation(records, use_whiten=False, part="all", n_jobs=-1, prefer='threads'):
    feats_per_layer = build_features(records, use_whiten=use_whiten, part=part)
    def score_li(li):
        data = feats_per_layer[li]
        if data is None:
            return np.nan
        X, y, bases = data
        if X.shape[0] < 9:
            return np.nan
        return lobo_score_for_layer(X, y, bases)
    scores = Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(score_li)(li) for li in range(L))
    return list(scores)

print("\n=== LOBO accuracy by layer (all features) ===")
acc_all = lobo_ablation(all_records, use_whiten=USE_WHITEN, part="all", n_jobs=os.cpu_count(), prefer='threads')
print(" ".join([f"L{i:02d}:{a:.3f}" if not np.isnan(a) else f"L{i:02d}:--" for i,a in enumerate(acc_all)]))

print("\n=== Ablations ===")
acc_geom = lobo_ablation(all_records, use_whiten=USE_WHITEN, part="geom", n_jobs=os.cpu_count(), prefer='threads')
acc_mom  = lobo_ablation(all_records, use_whiten=USE_WHITEN, part="mom",  n_jobs=os.cpu_count(), prefer='threads')
acc_spec = lobo_ablation(all_records, use_whiten=USE_WHITEN, part="spec", n_jobs=os.cpu_count(), prefer='threads')
np.save(OUTDIR / "lobo_all.npy", np.array(acc_all))
np.save(OUTDIR / "lobo_geom.npy", np.array(acc_geom))
np.save(OUTDIR / "lobo_mom.npy",  np.array(acc_mom))
np.save(OUTDIR / "lobo_spec.npy", np.array(acc_spec))

# -----------------------------
# 8) Per-token heatmaps (mean step/curv over tokens) for a few layers
# -----------------------------
def heatmap_tokenwise(layers=(4,13,21)):
    for li in layers:
        steps_by_pos=[]; curv_by_pos=[]
        for rec in all_records:
            Hc = cont_layer(rec, li)
            if Hc.shape[0] < 6: continue
            Hhat = l2_normalize_rows(whiten_tokens(Hc) if USE_WHITEN else Hc)
            d = step_dists(Hhat); dd = curvature(Hhat)
            steps_by_pos.append(d); curv_by_pos.append(dd)
        if not steps_by_pos: continue
        # align by min length
        m = min(map(len, steps_by_pos))
        S = np.stack([s[:m] for s in steps_by_pos], 0).mean(0)
        C = np.stack([c[:max(1,m-1)] for c in curv_by_pos], 0).mean(0)
        plt.figure(figsize=(6,2.4)); plt.plot(S); plt.title(f"Mean step vs token @ L{li}"); plt.xlabel("token step"); plt.ylabel("|Δ|"); plt.tight_layout()
        plt.savefig(OUTDIR / f"mean_step_L{li:02d}.png", dpi=140); plt.close()
        plt.figure(figsize=(6,2.4)); plt.plot(C); plt.title(f"Mean curvature vs token @ L{li}"); plt.xlabel("token index"); plt.ylabel("|Δ²|"); plt.tight_layout()
        plt.savefig(OUTDIR / f"mean_curv_L{li:02d}.png", dpi=140); plt.close()

heatmap_tokenwise(layers=(4,13,21,26))

# -----------------------------
# 9) Tiny bootstrap for stability (same as before, but save)
# -----------------------------
def rec_base_id(r): return rec_base(r["id"])
def lobo_probe(records, use_whiten=False):
    lab_map = get_lab_map(records)
    per_layer_scores = []
    for li in range(L):
        feats=[]; y_list=[]; bases_list=[]
        for rec in records:
            Hc = cont_layer(rec, li)
            pf = path_features(Hc, use_whiten=use_whiten)
            if pf is None: continue
            vec = np.concatenate([pf["geom"], pf["mom"], pf["spec"]])
            feats.append(vec); y_list.append(lab_map[rec["perspective"]]); bases_list.append(rec_base(rec["id"]))
        if len(feats) < 9: 
            per_layer_scores.append(np.nan); continue
        X = np.stack(feats); y = np.array(y_list); bases = np.array(bases_list)
        uniq = sorted(set(bases.tolist()))
        correct=0; total=0
        for b in uniq:
            train = bases != b; test = bases == b
            if test.sum()==0 or train.sum()<6: continue
            clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                                RidgeClassifierCV(alphas=np.logspace(-3,3,9)))
            clf.fit(X[train], y[train]); pred = clf.predict(X[test])
            correct += (pred==y[test]).sum(); total += test.sum()
        per_layer_scores.append(correct/total if total>0 else np.nan)
    return per_layer_scores

def bootstrap_lobo(records, B=30, n_jobs=-1, prefer='threads'):
    # Precompute features once for 'all' part (geom+mom+spec)
    feats_per_layer = build_features(records, use_whiten=USE_WHITEN, part='all')
    base_ids = sorted({rec_base_id(r) for r in records})
    per_layer_mu, per_layer_sd = [], []

    def one_boot(li, seed):
        data = feats_per_layer[li]
        if data is None:
            return np.nan
        X, y, bases = data
        rng = np.random.default_rng(seed)
        choose = rng.choice(base_ids, size=len(base_ids), replace=True)
        idxs = []
        for b in choose:
            idxs.extend(np.where(bases == b)[0].tolist())
        if not idxs:
            return np.nan
        return lobo_score_for_layer(X[idxs], y[idxs], bases[idxs])

    for li in range(L):
        seeds = [SEED + 1000*li + j for j in range(B)]
        scores = Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(one_boot)(li, sd) for sd in seeds)
        scores = [s for s in scores if not (isinstance(s, float) and np.isnan(s))]
        per_layer_mu.append(np.mean(scores) if scores else np.nan)
        per_layer_sd.append(np.std(scores) if scores else np.nan)
    return per_layer_mu, per_layer_sd

print("\nBootstrapping LOBO (B=30)...")
mu, sd = bootstrap_lobo(all_records, B=30, n_jobs=os.cpu_count(), prefer='threads')
for i,(m,s) in enumerate(zip(mu,sd)):
    if np.isnan(m): print(f"L{i:02d}: (insufficient)")
    else: print(f"L{i:02d}: mean={m:.3f} ± {s:.3f}")
np.save(OUTDIR / "lobo_bootstrap_mean.npy", np.array(mu))
np.save(OUTDIR / "lobo_bootstrap_sd.npy",   np.array(sd))

# -----------------------------
# 10) Simple layer-band report
# -----------------------------
def summarize_bands(geom_rows, align_rows, indices, bands=BANDS):
    def pick(arr, rng): 
        xs = [arr[i] for i in rng if i < len(arr)]
        return float(np.nanmean(xs)) if xs else np.nan
    dlt = [r["dlt_mean"] for r in geom_rows]
    cur = [r["curv_mean"] for r in geom_rows]
    low = [r["lowfreq_ratio"] for r in geom_rows]
    div = [r["delta"] for r in align_rows]
    CI  = [r["corridor_index"] for r in indices]
    DI  = [r["divergence_index"] for r in indices]
    out={}
    for name, rng in bands.items():
        out[name] = dict(
            dlt=pick(dlt, rng),
            curv=pick(cur, rng),
            lowE=pick(low, rng),
            tangent_divergence=pick(div, rng),
            corridor_index=pick(CI, rng),
            divergence_index=pick(DI, rng),
        )
    return out

band_report = summarize_bands(metrics_by_layer, align, indices, bands=BANDS)
with open(OUTDIR / "band_summary.json", "w") as f:
    _json.dump(band_report, f, indent=2)

print("\nBand summary:", band_report)
print("\nDone.")
