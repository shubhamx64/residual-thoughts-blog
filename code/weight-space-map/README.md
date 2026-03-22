# Weight-Space Map of Attention Heads in Gemma-2 (with SAEs)

> **Code snapshot** accompanying the blog post:
> [A Weight-Space Map of Attention Heads in Gemma-2 (with SAEs)](https://www.residual-thoughts.com/posts/weight-space-map-of-attention-heads/)
>
> This is a snapshot of experimental research code, not a maintained package.
> It is provided for transparency and reproducibility of the results described in the post.

## What This Does

Maps all 200 attention heads in Gemma-2-2B (layers 1-25, 8 heads per layer) using weight-space analysis with Sparse Autoencoder (SAE) decoder directions -- no activation sampling required for the core analysis.

### Pipeline Flow

```
main.py (CLI) --> analysis_pipeline.run_full_analysis()
  Per-layer --> run_layer_analysis():
    weight_extraction  --> W_Q, W_K, W_V, W_O (with RMSNorm gamma folding)
    sae_utils          --> SAE decoder directions, Gram matrices
    qk_routing         --> Phase 1: attention affinity B matrix, selectivity
    ov_writing         --> Phase 2: write vector semantics, copy vs transform
    feature_programs   --> Phase 3: triplet composition (query->key->write)
    rope_utils         --> Phase 4: RoPE position modulation curves
  synthesis            --> cross-layer patterns, steering recommendations
```

Validation (`activation_validation.py`, `run_validation.py`) hooks real attention logits and compares against weight-space predictions across distance bins.

Intervention (`run_intervention.py`) runs causal head ablation on factual retrieval tasks.

## Key Model Constants

| Constant | Value |
|----------|-------|
| Model | `google/gemma-2-2b` |
| Layers | 26 (layers 0-25) |
| Query heads | 8 per layer |
| KV heads | 4 per layer (GQA, 2:1 sharing) |
| Head dimension | 256 |
| SAE | `gemma-scope-2b-pt-res-canonical` @ `width_16k/canonical` (16,384 features) |
| SAE tap point | Post-MLP residual stream (layer offset -1) |
| Attention scaling | 256^{-0.5}, tanh soft-cap at 50.0 |
| Sliding window | 4096 on even layers, global on odd layers |
| RoPE | theta=10000, applied post-projection |

## Quick Start

### Prerequisites

- Python 3.10+
- GPU with sufficient VRAM (16GB+ recommended)
- HuggingFace account with access to [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) (gated model -- request access first)

### Install

```bash
pip install -r requirements.txt
huggingface-cli login
```

### Run

```bash
# Full analysis (all 26 layers, 16K SAE features)
python main.py --layers 0,5,10,15,20,25

# Quick mode (6 representative layers)
python main.py --quick

# Faster with fewer SAE features
python main.py --quick --subset-size 2048

# Activation validation
python run_validation.py --all-heads --layers 0,1,2,3,4,5

# Causal intervention
python run_intervention.py --run_all

# Generate blog figures (requires analysis + validation outputs)
python generate_blog_assets.py
```

## File Inventory

### Core Pipeline
| File | Description |
|------|-------------|
| `config.py` | Model architecture constants, SAE configuration, GQA head mapping |
| `weight_extraction.py` | Extract W_Q, W_K, W_V, W_O with RMSNorm gamma folding |
| `sae_utils.py` | Load SAEs via sae-lens, normalize decoders, compute Gram matrices |
| `qk_routing.py` | QK affinity matrix B, selectivity, diagonal dominance, baselines |
| `ov_writing.py` | OV write semantics: copy, transform, broadcast, suppress scores |
| `feature_programs.py` | Triplet composition (i->j->k), motif taxonomy |
| `rope_utils.py` | RoPE rotation matrices, stability curves across positions |
| `analysis_pipeline.py` | Orchestrator: runs phases 1-4 per layer |
| `synthesis.py` | Cross-layer pattern detection, steering recommendations |
| `main.py` | CLI entry point |

### Validation & Intervention
| File | Description |
|------|-------------|
| `activation_validation.py` | Hook real attention logits, correlate with weight-space predictions |
| `run_validation.py` | Validation runner with layer/head selection |
| `head_ablation.py` | Zero-ablation of head outputs via o_proj slicing |
| `run_intervention.py` | Causal intervention on factual retrieval tasks |
| `intervention_prompts.py` | Generate table-lookup retrieval prompts |

### Analysis & Figures
| File | Description |
|------|-------------|
| `generate_blog_assets.py` | Generate all figures from the blog post |
| `analyze_intervention.py` | Process intervention results, accuracy tables |
| `analyze_paired.py` | Paired statistics, Wilcoxon tests |
| `summarize_validation_report.py` | Parse validation logs into structured summaries |

### Data
| File | Description |
|------|-------------|
| `validation_prompts.csv` | Diverse prompts used for activation validation |
| `validation_prompts_short.csv` | Subset for quick testing |
