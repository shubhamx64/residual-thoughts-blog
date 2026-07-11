"""Streaming footprint sensors: MLP post-activation firing + residual-stream PR.

Never stores raw activations. Two modes:
  calibrate(texts)  -> per-layer abs-activation quantile thresholds
  capture(text)     -> per-layer sparse firing counts (per threshold) + PR stats
"""
import numpy as np
import torch

from common import find_mlp_down_projs, participation_ratio, SKIP_TOKENS, MAX_TOKENS, THRESH_QUANTILES


class FootprintSensor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.down_projs = find_mlp_down_projs(model)
        self.n_layers = len(self.down_projs)
        self.inter = self.down_projs[0][1].in_features
        self._buf = {}          # layer -> post-activation tensor for current forward
        self._handles = []
        self.thresholds = None  # {quantile: np.array[n_layers]}

    def _hook(self, layer_idx):
        def fn(module, inputs):
            self._buf[layer_idx] = inputs[0].detach()
        return fn

    def attach(self):
        for i, mod in self.down_projs:
            self._handles.append(mod.register_forward_pre_hook(self._hook(i)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _forward(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"].to(self.device)
        self._buf = {}
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True, use_cache=False)
        return ids, out

    def calibrate(self, texts, tokens_stride=4):
        """Per-layer abs-activation quantiles over a class-balanced calibration set."""
        samples = [[] for _ in range(self.n_layers)]
        for text in texts:
            _, _ = self._forward(text)
            for l in range(self.n_layers):
                a = self._buf[l][0, SKIP_TOKENS::tokens_stride].abs().float()
                # subsample neurons too: quantiles need volume, not completeness
                samples[l].append(a.flatten()[:: 7].cpu().numpy())
        self.thresholds = {}
        for q in THRESH_QUANTILES:
            self.thresholds[q] = np.array(
                [np.percentile(np.concatenate(samples[l]), q) for l in range(self.n_layers)],
                dtype=np.float32,
            )
        return self.thresholds

    def capture(self, text):
        """Returns dict with sparse per-layer firing counts and PR summary."""
        assert self.thresholds is not None, "calibrate() first"
        ids, out = self._forward(text)
        n_tok = ids.shape[1] - SKIP_TOKENS
        if n_tok < 32:
            return None
        rec = {"n_tokens": int(n_tok)}
        # token unigram counts: the surface-statistics baseline every footprint
        # metric must beat to claim regime (not tokenizer) structure
        tok_ids, tok_cnt = np.unique(ids[0, SKIP_TOKENS:].cpu().numpy(), return_counts=True)
        rec["tok_idx"] = tok_ids.astype(np.int32)
        rec["tok_cnt"] = tok_cnt.astype(np.int32)
        # PR over residual stream (block outputs), skipping early tokens
        pr_mean, pr_std = [], []
        for h in out.hidden_states[1:]:
            pr = participation_ratio(h[0, SKIP_TOKENS:])
            pr_mean.append(pr.mean().item())
            pr_std.append(pr.std().item())
        rec["pr_mean"] = np.array(pr_mean, dtype=np.float32)
        rec["pr_std"] = np.array(pr_std, dtype=np.float32)
        for l in range(self.n_layers):
            a = self._buf[l][0, SKIP_TOKENS:].abs()
            for q in THRESH_QUANTILES:
                thr = float(self.thresholds[q][l])
                counts = (a > thr).sum(0).to(torch.int32).cpu().numpy()
                nz = np.nonzero(counts)[0].astype(np.int32)
                rec[f"idx_q{q}_L{l}"] = nz
                rec[f"cnt_q{q}_L{l}"] = counts[nz]
        self._buf = {}
        return rec
