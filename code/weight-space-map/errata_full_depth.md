# Errata and full-depth revision: weight-space map of Gemma-2-2B attention heads

Working notes for the blog errata, with full 26-layer numbers from the corrected
pipeline (June 2026). Source runs: `analysis_20260610_233407.json` (analysis map,
4096 features, one_plus_gamma folding), `validation_full_sae.txt` /
`validation_full_token.txt` (20 prompts, all heads, RoPE-aware),
`validation_L13_offset0.txt`. Pre-fix baselines: `analysis_20260102_052006.json`,
`text_outputs/layers_all_4k_20_prompts.txt`.

## The three bugs

1. **RMSNorm gamma folding missed the +1.** HF `Gemma2RMSNorm` computes
   `y = (x / rms) * (1 + gamma)`; the pipeline folded `W * gamma` instead of
   `W * (1 + gamma)`. Since Gemma's learned gammas are small, this multiplied
   every folded weight by roughly gamma instead of ~1, distorting all published
   B matrices and selectivity numbers. Bug magnitude: per-layer cosine between
   legacy and corrected B matrices ranges 0.71 (L0) to 0.97.
2. **RoPE pairing convention.** The code rotated interleaved pairs (2i, 2i+1);
   HF Gemma-2 uses rotate_half pairs (i, i+128). Affected all RoPE stability
   curves and distance-binned validation. Keys must be rotated by -delta.
3. **Validation encoded the wrong activations.** Predictions were compared
   against SAE encodings of post-layernorm activations; Gemma Scope residual
   SAEs are trained on the raw residual. Fixed to encode raw block input with
   per-token 1/rms scaling.

## Retraction: the late-layer anti-predictive regime ("Regime 3") was an artifact

Per-layer mean Spearman (weight-space predicted attention vs real attention,
all 8 heads, 20 prompts):

| layer | pre-fix | fixed (SAE) | fixed (token) |
|---|---|---|---|
| 0 | - | 0.286 | 0.109 |
| 1 | 0.051 | 0.062 | 0.093 |
| 2 | 0.039 | 0.073 | -0.007 |
| 3 | 0.064 | 0.082 | 0.130 |
| 4 | -0.032 | 0.115 | 0.074 |
| 5 | 0.143 | 0.289 | 0.176 |
| 6 | 0.101 | 0.254 | 0.227 |
| 7 | -0.009 | 0.141 | 0.318 |
| 8 | 0.145 | 0.274 | 0.249 |
| 9 | -0.156 | 0.255 | 0.375 |
| 10 | -0.193 | 0.164 | 0.351 |
| 11 | 0.175 | 0.352 | 0.401 |
| 12 | -0.184 | 0.311 | 0.329 |
| 13 | -0.248 | 0.022 | 0.377 |
| 14 | 0.123 | 0.362 | 0.336 |
| 15 | -0.055 | 0.186 | 0.261 |
| 16 | -0.113 | 0.238 | 0.379 |
| 17 | 0.113 | 0.329 | 0.363 |
| 18 | 0.006 | 0.314 | 0.352 |
| 19 | 0.020 | 0.178 | 0.277 |
| 20 | -0.000 | 0.202 | 0.294 |
| 21 | 0.077 | 0.253 | 0.217 |
| 22 | 0.030 | 0.316 | 0.179 |
| 23 | -0.007 | 0.340 | 0.298 |
| 24 | -0.064 | 0.334 | 0.327 |
| 25 | 0.244 | 0.220 | 0.288 |

Grand means: pre-fix +0.011 (noise), fixed SAE +0.229, fixed token +0.260.
Every layer the blog called anti-predictive (L9, L10, L12, L13, L16, L24) is
solidly positive after the fixes. There is no anti-predictive regime. Best
single heads reach Spearman 0.50 overall (L11H3) and 0.63 on local distances
(L6H0), with 100% sign stability.

## Revision: the Layer-6 selectivity spike survives, its uniqueness does not

Corrected Sel x U (top-1 softmax mass over uniform): L6 is still the global
peak at 1451, but L8 (1176) and L22 (1113) are close behind, with L12 (866) and
L14 (912) in the same band. Pre-fix, L6 (106) appeared to tower 2-2.5x over
everything else; that contrast was the artifact. The corrected story is a band
of selective layers from L6 to L14 plus a late peak at L22, all 3-25x above a
random-weights baseline.

The diagonal-mass (identity/copy routing) peak moves from L6 to **L10**
(0.124), with L8 and L12 high. This matches an independent weight-space
finding: 2-hop composition chains relay through L8-L11 into L12/L13 readers.

RoPE stability AUCs rise everywhere under the corrected rotation convention
(0.63-0.73 pre-fix to 0.65-0.89 fixed; peaks L22 = 0.89, L14 = 0.88, L8 = 0.87).

## New: the L13 anomaly is a Gemma Scope checkpoint issue, not a model issue

L13 is the only layer where the corrected SAE-basis validation fails
(mean Spearman 0.022). Three probes of the same layer:

| probe | mean Spearman (8 heads) |
|---|---|
| layer-12 SAE (standard -1 offset, conceptually correct tap) | +0.022 |
| layer-13 SAE (offset 0, conceptually misaligned tap) | +0.176 |
| token-embedding basis (no SAE) | +0.377 |

The conceptually correct SAE is the worst of the three, and a dictionary-free
basis is the best. Attention at L13 is well-predicted by weight space; the
layer-12 Gemma Scope canonical 16k checkpoint is simply a poor dictionary for
this purpose. Any SAE-based analysis that touches gemma-scope-2b-pt-res
layer 12 should be treated with suspicion.

## Revision: the L23 ablation story

The causal finding stands: ablating L23H0 improves factual-retrieval accuracy
by +6.2 points over baseline (0.595 to 0.657, 800 prompts; L23H2 +3.2;
replicated across runs). The blog's explanation does not: it tied the
improvement to the late-layer anti-predictive regime, which was an artifact.
Under the corrected map, L23H0 is well-predicted (overall Spearman 0.32, local
0.57), so "the weight map can't see this head" is wrong.

What the corrected map actually says about L23H0: a REPULSION routing
archetype (strong negative affinities, min -37) combined with a high
copy-dominance OV circuit (copy_dominance 0.95). A head that copies content
under repulsion-shaped routing plausibly drags non-answer context into the
final-layer residual on cluttered prompts, which is consistent with ablation
helping on a filler-text retrieval task. Prediction quality and causal
helpfulness are orthogonal properties; the errata should decouple them.
Independent corroboration that L23 heads are special: instruction tuning moves
L23H0/H1 more than almost any other heads in the model (base vs -it diff).

## Dual-basis result (new, worth its own section in the blog)

The mean-centered token-embedding matrix, used as a probe basis with no SAE at
all, validates slightly better than Gemma Scope SAEs overall (grand mean 0.260
vs 0.229) and wins 15/26 layers, including nearly everything from L7 to L20.
The SAE basis wins at L0, the L5-L6 band, and L21-L24. Practical reading: for
routing analysis, SAE dictionaries are not load-bearing; for mid-stack layers
a free basis is better, and basis disagreement (as at L13) is a diagnostic for
SAE checkpoint quality.

## Pipeline health note

21.6% of feature programs in the full-depth run used fallback self-write
injection; the explicit-only program histogram is the trustworthy one for
circuit claims.
