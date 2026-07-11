# =========================
# Gemma-2-2B QK-in-SAE-space routing stats + proper controls (self-contained)
# Controls added:
#   (1) Independent random bases for Q and K  -> kills diagonal shortcut
#   (2) Permuted-K baseline using REAL SAE basis -> kills feature identity, preserves distribution
# Includes optional HF + Neuronpedia key prompts.
# =========================

#!pip -q install -U transformers accelerate safetensors sentencepiece huggingface_hub sae-lens neuronpedia

import os, re, json, math, functools, random
from getpass import getpass

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login as hf_login
from sae_lens import SAE
from neuronpedia.np_sae_feature import SAEFeature

# =========================
# MACROS (edit these)
# =========================
MODEL_ID = "google/gemma-2-2b"
L_ATTN_LIST = [0, 10, 20]             # attention layers

Q_HEAD_PAIR = (0, 1)                  # must share KV head (0,1) or (2,3) etc.

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH_ID = "width_16k/canonical"

M_SUBFEATS = 2048
N_ROWS = 256
TOPK = 20
K_PAIRS = 10

SEED = 42

DO_HF_LOGIN = True
DO_NEURONPEDIA = True
NP_CACHE_PATH = "neuronpedia_cache.json"  # safe on Colab; ignored on local if path differs

PRINT_TOP_PAIRS = 8
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| gpu:", torch.cuda.get_device_name(0) if device == "cuda" else None)

# ---- HF login (optional) ----
if DO_HF_LOGIN:
    try:
        tok = getpass("HF token (press Enter to skip): ")
        if tok.strip():
            hf_login(token=tok.strip())
            print("HF login: ok")
        else:
            print("HF login: skipped")
    except Exception as e:
        print("HF login: failed/skipped:", repr(e))

# ---- Neuronpedia key (optional) ----
if DO_NEURONPEDIA:
    try:
        if not os.getenv("NEURONPEDIA_API_KEY"):
            k = getpass("Neuronpedia API key (press Enter to skip): ")
            if k.strip():
                os.environ["NEURONPEDIA_API_KEY"] = k.strip()
                print("Neuronpedia key: set")
            else:
                print("Neuronpedia key: skipped")
    except Exception as e:
        print("Neuronpedia key: failed/skipped:", repr(e))

# ---- Load model + tokenizer ----
cfg = AutoConfig.from_pretrained(MODEL_ID)
tc = getattr(cfg, "text_config", cfg)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

n_q = tc.num_attention_heads
n_kv = tc.num_key_value_heads
d_h = tc.head_dim
group_size = n_q // n_kv
q_to_kv = [qh // group_size for qh in range(n_q)]
print("n_layers:", tc.num_hidden_layers, "n_q:", n_q, "n_kv:", n_kv, "d_h:", d_h, "group_size:", group_size)
print("q_to_kv:", q_to_kv)

q0, q1 = Q_HEAD_PAIR
assert q_to_kv[q0] == q_to_kv[q1], "Q_HEAD_PAIR must share the same KV head"

def detok(token_list):
    s = tokenizer.convert_tokens_to_string(token_list)
    return s.replace("<bos>", "").strip()

# ---- Neuronpedia disk cache to avoid re-fetching ----
_np_disk_cache = {}
if os.path.exists(NP_CACHE_PATH):
    try:
        with open(NP_CACHE_PATH, "r") as f:
            _np_disk_cache = json.load(f)
    except Exception:
        _np_disk_cache = {}

def _np_key(model_id, source, index): return f"{model_id}|{source}|{int(index)}"

@functools.lru_cache(maxsize=8192)
def np_payload(model_id: str, source: str, index: int) -> dict:
    key = _np_key(model_id, source, index)
    if key in _np_disk_cache:
        return _np_disk_cache[key]
    f = SAEFeature.get(model_id, source, str(int(index)))
    jd = getattr(f, "jsonData", None) or getattr(f, "json_data", None)
    if isinstance(jd, str):
        try: payload = json.loads(jd)
        except Exception: payload = {"_raw_jsonData": jd}
    elif isinstance(jd, dict):
        payload = jd
    else:
        payload = getattr(f, "__dict__", {}) or {}
    _np_disk_cache[key] = payload
    return payload

def flush_np_cache():
    try:
        with open(NP_CACHE_PATH, "w") as f:
            json.dump(_np_disk_cache, f)
        print("Saved Neuronpedia cache:", NP_CACHE_PATH, "| entries:", len(_np_disk_cache))
    except Exception as e:
        print("Failed to save Neuronpedia cache:", repr(e))

def evidence_label(model_id, source, index, k=4, n_snips=1, max_chars=80):
    p = np_payload(model_id, source, int(index))
    ps = p.get("pos_str") or []
    pv = p.get("pos_values") or []
    top_strs = [s for s,_ in list(zip(ps, pv))[:k]] if (ps and pv) else ps[:k]
    snips = []
    for ex in (p.get("activations") or [])[:n_snips]:
        toks = ex.get("token") or ex.get("tokens") or []
        if toks:
            txt = detok(toks).replace("\n", " ").strip()
            snips.append((txt[:max_chars] + "...") if len(txt) > max_chars else txt)
    return {"top_pos_str": top_strs, "snips": snips}

def is_format_style(blob: str) -> bool:
    pat = r"(\\math|\\ref|\\kappa|\\emptyset|{|\}|;|\(|\)|->|SELECT|public|class|#include|udp->|Wireshark|\.php|ſ|<div|</|<\/|::|\$\w+)"
    return re.search(pat, blob) is not None

def tag_feature(model_id, source, index) -> str:
    ev = evidence_label(model_id, source, int(index), k=6, n_snips=1, max_chars=120)
    blob = " ".join(ev["top_pos_str"] + ev["snips"])
    return "format/style" if is_format_style(blob) else "maybe-semantic"

def compute_metrics(Qf, Kf, rows, topk):
    Qn = F.normalize(Qf.float(), dim=1)
    Kn = F.normalize(Kf.float(), dim=1)
    S = (Qn[rows] @ Kn.T).float()   # [n_rows, m]
    topv, topi = torch.topk(S, k=topk, dim=1)
    gap = (topv[:,0] - topv[:,1]).detach().cpu()
    top1 = topv[:,0].detach().cpu()
    diag = (topi[:,0].detach().cpu() == rows.detach().cpu()).float().mean().item()
    return {
        "top1_mean": float(top1.mean().item()),
        "top1_max": float(top1.max().item()),
        "gap_median": float(gap.median().item()),
        "gap_max": float(gap.max().item()),
        "diag_top1_rate": float(diag),
    }, topv.detach().cpu(), topi.detach().cpu()

def top_pairs(topv, topi, rows, feat_idx, k_pairs):
    topk = topi.shape[1]
    flat_scores = topv.reshape(-1)
    flat_q = rows.repeat_interleave(topk)
    flat_k = topi.reshape(-1)
    k_pairs = min(k_pairs, flat_scores.numel())
    bestv, bestpos = torch.topk(flat_scores, k=k_pairs)
    best_q_sub = flat_q[bestpos].tolist()
    best_k_sub = flat_k[bestpos].tolist()
    best_s = bestv.tolist()
    best_q_global = [int(feat_idx[q].item()) for q in best_q_sub]
    best_k_global = [int(feat_idx[k].item()) for k in best_k_sub]
    return list(zip(best_q_global, best_k_global, best_s))

def run_layer(L_attn: int):
    assert 0 <= L_attn < tc.num_hidden_layers
    L_sae = max(L_attn - 1, 0)

    # SAE basis
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"layer_{L_sae}/{SAE_WIDTH_ID}")
    D = sae.W_dec.detach().cpu()  # [16384, d_model]

    g = torch.Generator().manual_seed(SEED + 1000*L_attn + 17)
    feat_idx = torch.randperm(D.shape[0], generator=g)[:M_SUBFEATS]
    Dsub = F.normalize(D[feat_idx], dim=1)           # [m, d_model]

    # attention weights
    layer = model.model.layers[L_attn]
    attn = layer.self_attn
    gamma = layer.input_layernorm.weight.detach().cpu()

    Wq = attn.q_proj.weight.detach().cpu().view(n_q,  d_h, -1)
    Wk = attn.k_proj.weight.detach().cpu().view(n_kv, d_h, -1)
    Wq_eff = Wq * gamma.view(1,1,-1)
    Wk_eff = Wk * gamma.view(1,1,-1)

    kvh = q_to_kv[q0]
    Wk_ = Wk_eff[kvh].to(device=device, dtype=torch.float16)

    # sampled rows
    n_rows = min(N_ROWS, M_SUBFEATS)
    rows = torch.randperm(M_SUBFEATS, generator=g)[:n_rows].cpu()

    Dsub_ = Dsub.to(device=device, dtype=torch.float16)

    # REAL Kf
    Kf_real = Dsub_ @ Wk_.T

    # PERMUTED-K baseline: same Dsub for Q, but K uses permuted rows
    perm = torch.randperm(M_SUBFEATS, generator=torch.Generator().manual_seed(SEED + 777 + 1000*L_attn))
    Dsub_perm_ = Dsub_[perm]
    Kf_perm = Dsub_perm_ @ Wk_.T

    # INDEPENDENT-RANDOM baseline: Q and K use different random bases
    gq = torch.Generator().manual_seed(SEED + 9001 + 1000*L_attn)
    gk = torch.Generator().manual_seed(SEED + 9002 + 1000*L_attn)
    Dq = F.normalize(torch.randn((M_SUBFEATS, D.shape[1]), generator=gq), dim=1).to(device=device, dtype=torch.float16)
    Dk = F.normalize(torch.randn((M_SUBFEATS, D.shape[1]), generator=gk), dim=1).to(device=device, dtype=torch.float16)
    Kf_ind = Dk @ Wk_.T

    out = {"L_attn": L_attn, "L_sae_basis": L_sae, "kv_head": kvh, "q_heads": [q0,q1]}

    head_out = {}
    topi_store = {}
    for qh in [q0, q1]:
        Wq_ = Wq_eff[qh].to(device=device, dtype=torch.float16)

        # REAL
        Qf_real = Dsub_ @ Wq_.T
        m_real, topv, topi = compute_metrics(Qf_real, Kf_real, rows.to(device), TOPK)
        pairs = top_pairs(topv, topi, rows, feat_idx, K_PAIRS)
        topi_store[qh] = topi

        # PERMUTED-K
        m_perm, _, _ = compute_metrics(Qf_real, Kf_perm, rows.to(device), TOPK)

        # INDEPENDENT-RANDOM
        Qf_ind = Dq @ Wq_.T
        m_ind, _, _ = compute_metrics(Qf_ind, Kf_ind, rows.to(device), TOPK)

        head_out[qh] = {
            "REAL": m_real,
            "PERMUTED_K": m_perm,
            "INDEP_RANDOM": m_ind,
            "top_pairs": pairs,
        }

    out["heads"] = head_out

    # within-group similarity on REAL topi
    t0, t1 = topi_store[q0], topi_store[q1]
    top1_match = (t0[:,0] == t1[:,0]).float().mean().item()
    overlaps = []
    for r in range(t0.shape[0]):
        overlaps.append(len(set(t0[r].tolist()) & set(t1[r].tolist())))
    out["within_group"] = {
        "top1_match_rate": float(top1_match),
        "mean_topk_overlap": float(sum(overlaps)/len(overlaps)),
        "topk": int(TOPK),
    }

    # Neuronpedia tagging (on top-pair features only)
    if DO_NEURONPEDIA and os.getenv("NEURONPEDIA_API_KEY"):
        NP_SOURCE = f"{L_sae}-gemmascope-res-16k"
        feat_set = set()
        for qh in [q0,q1]:
            for (qg, kg, _) in out["heads"][qh]["top_pairs"]:
                feat_set.add(int(qg)); feat_set.add(int(kg))
        tags = {f: tag_feature("gemma-2-2b", NP_SOURCE, f) for f in feat_set}
        out["neuronpedia"] = {
            "source": NP_SOURCE,
            "format_style_count": int(sum(t=="format/style" for t in tags.values())),
            "tagged_features_total": int(len(tags)),
        }

        # add observed snippets for printed pairs
        for qh in [q0,q1]:
            obs = []
            for (qg, kg, s) in out["heads"][qh]["top_pairs"][:PRINT_TOP_PAIRS]:
                ev_q = evidence_label("gemma-2-2b", NP_SOURCE, qg)
                ev_k = evidence_label("gemma-2-2b", NP_SOURCE, kg)
                obs.append({"q": int(qg), "k": int(kg), "cos": float(s), "q_obs": ev_q, "k_obs": ev_k})
            out["heads"][qh]["top_pairs_observed"] = obs

    return out

# -------------------------
# RUN
# -------------------------
results = []
for L in L_ATTN_LIST:
    r = run_layer(L)
    results.append(r)

    print("\n==============================")
    print(f"Layer attn={r['L_attn']} | SAE basis={r['L_sae_basis']} | KV head={r['kv_head']}")
    wg = r["within_group"]
    print("Within-group (REAL): top1_match_rate=", f"{wg['top1_match_rate']:.4f}",
          "mean_topk_overlap=", f"{wg['mean_topk_overlap']:.3f}", f"of {wg['topk']}")

    for qh in r["q_heads"]:
        h = r["heads"][qh]
        def fmt(m): return f"top1_mean/max={m['top1_mean']:.4f}/{m['top1_max']:.4f} gap_med/max={m['gap_median']:.4f}/{m['gap_max']:.4f} diag={m['diag_top1_rate']:.4f}"
        print(f"\nq_head={qh} REAL        {fmt(h['REAL'])}")
        print(f"q_head={qh} PERMUTED_K  {fmt(h['PERMUTED_K'])}")
        print(f"q_head={qh} INDEP_RAND  {fmt(h['INDEP_RANDOM'])}")

    if "neuronpedia" in r:
        np = r["neuronpedia"]
        print("\nNeuronpedia tagging:", f"format/style {np['format_style_count']} of {np['tagged_features_total']}",
              "| source:", np["source"])
    else:
        print("\nNeuronpedia tagging: skipped")

out_path = "gemma2_qk_sae_routing_stats_fixed_baselines.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", out_path)

if DO_NEURONPEDIA and os.getenv("NEURONPEDIA_API_KEY"):
    flush_np_cache()
