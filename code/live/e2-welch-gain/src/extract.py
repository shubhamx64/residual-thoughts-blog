"""Weight extraction with norm-gamma folding, per architecture family.

All dictionaries are expressed in residual-stream coordinates:
  reads  : rows of (W @ diag(g_pre))   -- pre-norm scale folded into the read
  writes : columns of (diag(g_post) @ W) -- post-norm scale folded into the write
Gemma-2 RMSNorm stores weight as an offset: effective scale is (1 + w), and it
has post-attention / post-feedforward norms that scale the block output before
the residual add. Pythia uses LayerNorm (bias and mean-subtraction ignored for
these geometry stats -- documented approximation) and parallel residual.
"""
import torch

MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "gemma-2-2b": "google/gemma-2-2b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
}


def load_weights(model_key, device="cuda"):
    from transformers import AutoModelForCausalLM
    dtype = torch.float32  # weights-only; exact spectra matter more than speed
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], dtype=dtype,
                                                 low_cpu_mem_usage=True)
    model.eval()
    return model


def _gemma_scale(w):
    return 1.0 + w.float()


def extract_layers(model, model_key):
    """Yield per-layer dicts of residual-coordinate matrices + head geometry."""
    cfg = model.config
    d = cfg.hidden_size
    out = []
    if "pythia" in model_key:
        n_heads = cfg.num_attention_heads
        head_dim = d // n_heads
        for layer in model.gpt_neox.layers:
            g_in = layer.input_layernorm.weight.float()
            g_post = layer.post_attention_layernorm.weight.float()
            qkv = layer.attention.query_key_value.weight.float()  # (3d, d) interleaved per head
            qkv = qkv.view(n_heads, 3, head_dim, d)
            Wq = (qkv[:, 0] * g_in).reshape(n_heads * head_dim, d)
            Wk = (qkv[:, 1] * g_in).reshape(n_heads * head_dim, d)
            Wv = (qkv[:, 2] * g_in).reshape(n_heads * head_dim, d)
            Wo = layer.attention.dense.weight.float()             # (d, n_heads*head_dim)
            # parallel residual: MLP reads the same pre-attn stream via its own norm
            g_mlp = layer.post_attention_layernorm.weight.float()
            Wup = layer.mlp.dense_h_to_4h.weight.float() * g_mlp  # (4d, d)
            Wdown = layer.mlp.dense_4h_to_h.weight.float()        # (d, 4d)
            out.append({"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                        "Wgate": Wup, "Wup": Wup, "Wdown": Wdown,
                        "n_heads": n_heads, "n_kv": n_heads, "head_dim": head_dim})
        return out, d

    layers = model.model.layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", d // n_heads)
    gemma = "gemma" in model_key
    for layer in layers:
        sc = _gemma_scale if gemma else (lambda w: w.float())
        g_attn_in = sc(layer.input_layernorm.weight)
        Wq = layer.self_attn.q_proj.weight.float() * g_attn_in
        Wk = layer.self_attn.k_proj.weight.float() * g_attn_in
        Wv = layer.self_attn.v_proj.weight.float() * g_attn_in
        Wo = layer.self_attn.o_proj.weight.float()
        if gemma:
            g_attn_out = sc(layer.post_attention_layernorm.weight)
            Wo = g_attn_out[:, None] * Wo
            g_mlp_in = sc(layer.pre_feedforward_layernorm.weight)
            g_mlp_out = sc(layer.post_feedforward_layernorm.weight)
        else:
            g_mlp_in = sc(layer.post_attention_layernorm.weight)
            g_mlp_out = None
        Wgate = layer.mlp.gate_proj.weight.float() * g_mlp_in
        Wup = layer.mlp.up_proj.weight.float() * g_mlp_in
        Wdown = layer.mlp.down_proj.weight.float()
        if g_mlp_out is not None:
            Wdown = g_mlp_out[:, None] * Wdown
        out.append({"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                    "Wgate": Wgate, "Wup": Wup, "Wdown": Wdown,
                    "n_heads": n_heads, "n_kv": n_kv, "head_dim": head_dim})
    return out, d


def embedding_matrix(model):
    return model.get_input_embeddings().weight.float()
