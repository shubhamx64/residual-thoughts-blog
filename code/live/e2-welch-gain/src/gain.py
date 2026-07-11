"""Gain map: spectral norms of weight matrices and per-head circuit gains.

sigma_QK^h = sigma_max((W_Q^h)^T W_K^h) / sqrt(head_dim)  -- routing sharpness
sigma_OV^h = sigma_max(W_O^h W_V^h)                       -- value-path amplification
Cumulative depth gain: prod (1 + g_attn_l)(1 + g_mlp_l) with
  g_attn = max_h sigma_OV^h (attention weights are row-stochastic, so the value
           path bounds the block gain), g_mlp = sigma_max(Wdown) * sigma_max(Wup).
HEURISTIC upper map, not a theorem: RMSNorm is not a fixed linear map and the
GeGLU product is not globally Lipschitz. Useful for relative comparisons only.
"""
import torch


@torch.no_grad()
def sigma_max(W, device="cuda"):
    return float(torch.linalg.matrix_norm(W.to(device), ord=2))


@torch.no_grad()
def layer_gains(L, device="cuda"):
    n_heads, n_kv, hd = L["n_heads"], L["n_kv"], L["head_dim"]
    group = n_heads // n_kv
    Wq = L["Wq"].to(device).view(n_heads, hd, -1)
    Wk = L["Wk"].to(device).view(n_kv, hd, -1)
    Wv = L["Wv"].to(device).view(n_kv, hd, -1)
    Wo = L["Wo"].to(device)                      # (d, n_heads*hd)
    d = Wo.shape[0]
    Wo_h = Wo.view(d, n_heads, hd).permute(1, 0, 2)  # (n_heads, d, hd)

    qk, ov = [], []
    for h in range(n_heads):
        kv = h // group
        A = Wq[h].T @ Wk[kv] / (hd ** 0.5)       # (d, d) rank <= hd
        qk.append(float(torch.linalg.matrix_norm(A, ord=2)))
        B = Wo_h[h] @ Wv[kv]                     # (d, d) rank <= hd
        ov.append(float(torch.linalg.matrix_norm(B, ord=2)))

    s_down = sigma_max(L["Wdown"], device)
    s_up = sigma_max(L["Wup"], device)
    s_gate = sigma_max(L["Wgate"], device)
    return {"qk": qk, "ov": ov,
            "sigma_down": s_down, "sigma_up": s_up, "sigma_gate": s_gate,
            "g_attn": max(ov), "g_mlp": s_down * s_up}
