"""Minimal GPTQ: Hessian-compensated quantization, group-wise scales, and two
protection mechanisms that map onto neuron channels:

  row_bits   (out_features,) long  -- per-output-row bit widths (gate/up rows)
  keep_cols  (in_features,) bool   -- input columns kept in full precision
                                      (down-proj neuron columns; SpQR-style)

Reference algorithm (Frantar et al. 2022): symmetric RTN grid, per-(row, group)
scales recomputed on error-compensated weights at each group boundary, no
act-order.
"""
import torch

GROUP = 128
PERCDAMP = 0.01


@torch.no_grad()
def gptq_quantize(W, H, bits=4, row_bits=None, keep_cols=None, group=GROUP):
    """W: (out, in) float32 CUDA; H: (in, in) float32 CUDA. Returns new W."""
    out_f, in_f = W.shape
    W = W.clone()
    H = H.clone()

    diag = torch.diag(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0

    damp = PERCDAMP * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    if row_bits is None:
        row_bits = torch.full((out_f,), bits, device=W.device, dtype=torch.long)
    row_qmax = (2.0 ** (row_bits.float() - 1) - 1).unsqueeze(1)   # (out, 1)
    if keep_cols is None:
        keep_cols = torch.zeros(in_f, dtype=torch.bool, device=W.device)

    Q = torch.zeros_like(W)
    for g0 in range(0, in_f, group):
        g1 = min(g0 + group, in_f)
        n = g1 - g0
        scale = (W[:, g0:g1].abs().amax(dim=1, keepdim=True) / row_qmax).clamp_min(1e-10)
        Err = torch.zeros(out_f, n, device=W.device)
        for i in range(n):
            col = g0 + i
            w = W[:, col]
            if keep_cols[col]:
                q = w
            else:
                q = (w / scale.squeeze(1)).round().clamp(
                    -row_qmax.squeeze(1) - 1, row_qmax.squeeze(1)) * scale.squeeze(1)
            Q[:, col] = q
            e = (w - q) / Hinv[col, col]
            if col + 1 < g1:
                W[:, col + 1:g1] -= e.unsqueeze(1) * Hinv[col, col + 1:g1].unsqueeze(0)
            Err[:, i] = e
        if g1 < in_f:
            W[:, g1:] -= Err @ Hinv[g0:g1, g1:]
    return Q
