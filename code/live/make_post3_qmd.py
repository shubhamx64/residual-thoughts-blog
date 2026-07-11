"""Emit the Quarto post (posts/price-of-packing/index.qmd) for the blog,
inlining the four census SVGs from make_post3. Figures are passed through as
raw-HTML blocks so they inherit nothing from the theme except width scaling;
all prose styling comes from the blog's own CSS.
"""
from pathlib import Path

from make_post3 import fig1, fig2, fig3, fig4  # SVG strings (no braces)

BLOG = Path(__file__).resolve().parent / "residual-thoughts-blog"
POST = BLOG / "posts" / "price-of-packing"


def figure(svg, caption):
    return ('```{=html}\n<figure style="margin:1.6rem 0;padding:0;">\n'
            + svg
            + f'\n<figcaption style="font-size:0.85rem;color:#6b7280;'
            f'line-height:1.5;margin-top:0.6rem;">{caption}</figcaption>\n'
            '</figure>\n```\n')


TEMPLATE = r'''---
title: "Polysemanticity Is Not the Price of Packing"
subtitle: "A four-family, two-interface, pre-registered null -- with a positive control to prove the test could have seen the law"
author: "Shubham Srivastava"
date: "2026-07-05"
categories: [mechanistic-interpretability, superposition, polysemanticity, mlp-neurons, negative-results]
description: "Superposition has a folk corollary: the neurons that mix meanings should be the ones whose weight vectors crowd together. Tested dictionary-free on both faces of the MLP across four model families, it is false at neuron granularity -- and a planted-law control shows the pipeline would have caught it if it were true."
toc-depth: 3
---

Superposition theory has a folk corollary that almost everyone reaches for at
some point: the neurons that **mix meanings** ought to be the ones whose weight
vectors **crowd together**, because crowding is what packing many features into
few dimensions looks like. It is a clean, testable claim, and if it held, weight
geometry alone would hand you a zero-data polysemanticity map -- find the crowded
neurons, you have found the mixed ones.

These are lab notes on testing that claim directly. Dictionary-free (no SAE),
pre-registered, on both faces of the MLP, across four model families spanning
gated and plain architectures. The short version: it is false at neuron
granularity, on both interfaces, in every family -- and because a null is only
worth as much as the instrument behind it, I built a positive control that
recovers the same law at $\rho = +0.52$ when the law is planted by construction.

::: {.callout-note appearance="simple"}
[**Code**](https://github.com/shubhamx64/residual-thoughts-blog/tree/main/code/price-of-packing) -- the census (`census.py`, `census_read.py`), the statistics (`stats_e5.py`), and the two toy controls (`planted_control.py`, `toy_control.py`). All quantities come from a ~4-minute forward capture per model plus the weights; no activations are stored.
:::

::: {.callout-tip}
## TL;DR
- **The claim:** in superposition, geometrically crowded neurons should be the polysemantic ones. Made measurable without a dictionary -- mixing as regime entropy, crowding as neighbourhood density -- and tested with a partial correlation that removes the firing-rate confound.
- **Write interface (output directions):** median partial $\rho$ between mixing and crowding is **+0.00 to +0.05** across four families. Null.
- **Read interface (input directions):** the last place a neuron-level law could hide, since recruitment is decided on the read side. **-0.005 to +0.033**. Also null.
- **The two interfaces disagree per neuron** ($\rho$ 0.21--0.47), so there is no single "this neuron is crowded" for mixing to track. That decorrelation is a finding, not a footnote.
- **The only above-noise effects flip sign between families** -- Gemma-2's isolated neurons are its cleanest, TinyLlama's are its most mixed. That is the signature of noise around zero, not a weak law.
- **Positive control:** the identical statistic recovers a *planted* crowding->mixing law at **+0.52**, and the coarse entropy proxy tracks ground-truth neuron mixing at **+0.83** in a trained toy. The instrument is not blind; there is simply nothing to see in the real models.
- **Scope:** neurons, not features; task-regime mixing, not fine semantics; MLP, not attention. This bounds a shortcut, it does not refute feature-level superposition.
:::

## The claim, made measurable without a dictionary

In superposition a network represents more features than it has dimensions by
letting feature directions overlap. The intuitive reading at the *neuron* level:
a neuron in a geometrically crowded neighbourhood is being asked to serve many
masters, so it should fire across many unrelated contexts -- it should be
polysemantic.

Both sides become concrete without ever touching an SAE. Mixing is the Shannon
entropy of a neuron's firing distribution over five task regimes (math,
math-prose, code, code-prose, prose), with per-class rates normalised by token
counts first:

$$H_i = -\sum_{c} p_{ic}\,\log_2 p_{ic}, \qquad p_{ic} = \frac{r_{ic}}{\sum_{c'} r_{ic'}}.$$

Crowding is the neighbourhood density of a neuron's weight vector -- the number
of same-layer neurons within $|\cos| > 0.4$ of it (with $\max|\cos|$ and the
mean of the top-10 neighbours as secondaries).

Every correlation below is a **partial** Spearman $\rho$, controlling for log
firing rate. A busy neuron is mechanically both more mixed and more likely to
look crowded; rank-residualising on rate before correlating removes that
confound. Neurons with fewer than 50 firing events are excluded. Seeds fixed;
four families -- Qwen2.5-1.5B, Gemma-2-2B, Pythia-1.4B, TinyLlama-1.1B -- chosen
so a Gemma-only quirk cannot masquerade as a law.

## Everything hugs zero -- on both faces of the neuron

::: {.callout-important}
## Verdict: null, 4/4 families, both interfaces
The median-across-layers partial $\rho$ between mixing and crowding is **+0.00 to +0.05 on the write interface** (down-projection columns) and **-0.005 to +0.033 on the read interface** (gate-projection rows) -- every one inside the pre-registered $|\rho| < 0.10$ null band. No couple-type stratum separates by more than 0.12 bits at matched firing rate. Robust across firing thresholds (q98/q99.5), a four-class entropy variant, and an outside-top-class proxy; the null *strengthens* when universal-neuron candidates are removed.
:::

__FIG1__

The write side was the natural place to look: a neuron's output direction is what
collides in the residual stream. But recruitment is decided on the *read* side --
whether a neuron fires is a property of its input weights, not its output. So the
read interface was the last place a neuron-level packing law could hide. It does
not hide there either. And the medians are not papering over a wide spread:

__FIG2__

## A neuron is not "in a crowded place." It is crowded per interface.

The read-side test was not a redundant re-run of the write-side test, and the
reason is a finding in itself. If a neuron simply occupied a crowded or an
uncrowded region of the model, its read-crowding and write-crowding would move
together -- the correlation would be near one. It is $\rho$ **0.21 to 0.47**.

__FIG4__

Recruitment geometry and expression geometry are substantially independent
properties of the same unit. The folk picture assumes one geometric fact per
neuron; there are at least two, they disagree, and neither predicts what the
neuron actually does across regimes. That is the quiet mechanism behind the whole
null.

## The effects that clear the noise point in opposite directions

A weak-but-real law would at least be *consistent*: crowded neurons a little more
mixed in every family. Instead the largest excursions cancel. Across both
interfaces the two measures that graze the band -- Gemma-2's write-side
local-crowding at $+0.12$ and Pythia's read-side top-10 at $-0.12$ -- have
opposite signs, one family each, two hundredths past a threshold chosen as the
noise floor. The stratum medians tell the same story in a picture.

__FIG3__

Gemma-2 showing the lone sub-threshold positive is exactly the pattern to expect:
across this whole line of work Gemma-2 is repeatedly the most weight-legible
model, so a whiff of structure-to-function signal there, and nowhere else, reads
as a property of Gemma rather than of transformers. It was pre-registered as not
counting.

## A null is only as good as the instrument behind it

The obvious attack on any negative result: maybe the pipeline could not detect
the effect even if it existed -- maybe five coarse regimes cannot resolve real
polysemanticity. That objection is answerable in-house, and cheaply, with two toy
systems.

::: {.callout-important}
## Positive control: pass, both flanks
**Detection.** On synthetic data with a graded crowding->mixing law planted by construction, the *identical* statistic recovers it at partial $\rho$ **+0.52** (density) / **+0.63** (top-10), while the same test on label-shuffled data stays at q95 $|\rho|$ 0.094 -- inside the null band.

**Proxy validity.** In a trained Toy-Models-style autoencoder where each neuron's true feature composition is known, the coarse five-regime entropy tracks ground-truth mixing at $\rho$ **+0.83**. The measure sees mixing; the pipeline sees the law when the law is there.
:::

| quantity | value | bar | reading |
|---|---|---|---|
| real census, write (best family) | +0.05 | -- | the null |
| planted law, density | +0.52 | $\geq$ +0.15 | detected, 3--4$\times$ over floor |
| planted law, top-10 | +0.63 | agrees in sign | detected |
| planted law, labels shuffled (q95) | 0.094 | < 0.10 | no false positive |
| census entropy vs ground-truth mixing | +0.83 | -- | proxy is faithful |

: The real-model numbers sit an order of magnitude below what the same instrument returns when the law is real. {tbl-colwidths="[46,14,18,22]"}

::: {.callout-note}
## An artifact worth naming
The first planted control used a clean two-group split (isolated vs crowded), which made the crowding variable perfectly bimodal. Pushed through the rank-residualisation, a near-binary regressor inflated the permutation null to q95 $\approx 0.16$ -- which would have made the instrument look *less* sensitive than it is. The fix was to regrade the planted data to a continuous crowding level matching the real census's density spread; the null fell back to $\approx 0.04$ and the detection signal was unchanged (bimodal $+0.83$, graded $+0.52$). The regrade fixed the *null estimate*, not the *signal* -- the kind of thing a control has to get right, and the kind of thing that is only visible if you keep both runs.
:::

::: {.callout-note}
## The mechanistic "why," seen in a toy
The trained autoencoder carries feature-level superposition by construction -- 2560 sparse features in 512 neurons, 5$\times$ compression, dense packing, clean reconstruction. Yet its neuron write-columns barely crowd at all: the maximum pairwise $|\cos|$ across the whole layer is **0.15**. Here is a minimal system where superposition is present and heavy, and it still does not imprint as neuron-level geometric crowding -- because features and neurons are simply different objects. That is the decoupling shown where you can see both levels at once.
:::

## What this does and does not say

This is a claim about **neurons**, not features. Superposition theory is
fundamentally about features as linear combinations of neurons; an SAE feature
can be crowded and polysemantic in ways no single neuron reveals. Nothing here
refutes feature-level superposition. What it bounds is the *neuron-granularity
shortcut* -- the hope that you could read polysemanticity off raw weight geometry
without a dictionary. You cannot, on either interface, in any of four families.

It is also a claim about **task-regime** mixing (five coarse classes), not
fine-grained semantic mixing, and about **MLP** geometry, not attention. Within
those bounds it is as airtight as I could make it: pre-registered thresholds,
both faces of the neuron, four architectures, a positive control at 3--4$\times$
the detection floor, and a demonstrated-faithful mixing proxy.

Which neurons mix regimes is set by the statistics of the data a model was
trained on, not by the geometric necessity of packing. The ledger of a model's
capacity has to be *read from usage*; it is not sitting latent in the Gram matrix
of its weights.

This is the negative half of a longer weights-first program. Where the
[earlier](../weight-space-map-of-attention-heads/)
[notes](../correcting-the-weight-space-map/) mapped what weight geometry *does*
know, this one marks a firm boundary on what it does not.
'''


def main():
    POST.mkdir(parents=True, exist_ok=True)
    doc = TEMPLATE
    doc = doc.replace("__FIG1__", figure(
        fig1,
        "Each point is the median-across-layers partial correlation for one "
        "model on one interface. Filled = write, open = read. The grey band is "
        "the pre-registered null; the dashed line is the +0.15 threshold a real "
        "effect had to clear. Eight of eight real measurements sit in the band. "
        "The red square is the same statistic run on synthetic data with a "
        "crowding-to-mixing law planted by construction -- it lands at +0.52."))
    doc = doc.replace("__FIG2__", figure(
        fig2,
        "The medians are not hiding a wide spread: every individual layer's "
        "partial &rho; (write and read density), all four families. The cloud "
        "sits on zero at every depth. The red dashed line marks where the "
        "planted-law control lands -- the entire real distribution is an order "
        "of magnitude short of it."))
    doc = doc.replace("__FIG4__", figure(
        fig4,
        "Per-neuron agreement between read-side and write-side neighbourhood "
        "density, per family. Recruitment geometry and expression geometry are "
        "substantially independent properties of the same unit -- so there is "
        "no single scalar &lsquo;this neuron is crowded.&rsquo;"))
    doc = doc.replace("__FIG3__", figure(
        fig3,
        "Rate-matched median mixing for a family's most-isolated vs "
        "most-crowded neurons. A universal law would lean every bar the same "
        "way. Gemma-2's isolated neurons are its cleanest; TinyLlama's are its "
        "most mixed -- the effect reverses between architectures."))
    (POST / "index.qmd").write_text(doc, encoding="utf-8")
    print(f"wrote {POST / 'index.qmd'} ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
