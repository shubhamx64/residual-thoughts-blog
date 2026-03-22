"""
Synthesis and Visualization for Gemma-2 Weight-Space Analysis.

Generates cross-layer summaries, depth-wise plots, and identifies
steering-optimal layer bands.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LayerSummary:
    """Summary metrics for a single layer."""
    layer_idx: int
    is_sliding_window: bool
    sae_layer_used: int
    
    # Routing metrics (averaged over heads)
    avg_diagonal_dominance: float
    avg_selectivity: float  # top1_mass_mean
    avg_diagonal_softmax_mass: float  # mean(softmax(B)[i,i]) - identity-sensitive
    avg_max_gap: float
    avg_asymmetry: float
    avg_effective_rank: float
    
    # Writing metrics (averaged over heads)
    avg_copy_score: float
    avg_transform_score: float
    avg_write_diversity: float
    
    # RoPE stability (averaged over heads)
    avg_semantic_controllability: float
    
    # Program distribution
    dominant_program_type: str
    program_type_entropy: float
    
    # Head diversity
    cross_head_redundancy: float
    
    # === Fields with defaults (must come last in dataclass) ===
    
    # Head-level dispersion (std across heads) - shows if structure is concentrated
    std_selectivity: float = 0.0
    std_diagonal_softmax_mass: float = 0.0
    
    # Write quality diagnostics
    median_top_write_similarity: float = 0.0  # Diagnoses weak SUPPRESS classifications
    
    # Baseline metrics (averaged over heads)
    baseline_rand_weights_selectivity: float = 0.0
    baseline_permuted_k_diag_mass: float = 0.0
    baseline_indep_bases_diagdom: float = 0.0


@dataclass
class DepthPattern:
    """Patterns identified across depth."""
    # Which layers are best for steering?
    high_controllability_layers: List[int]
    high_selectivity_layers: List[int]
    low_redundancy_layers: List[int]
    
    # Recommended steering band
    steering_band_start: int
    steering_band_end: int
    steering_band_score: float
    
    # Depth trends
    controllability_trend: str  # "increasing", "decreasing", "peaked", "flat"
    selectivity_trend: str
    copy_vs_transform_trend: str


def _get(obj, *keys):
    """Helper to get nested attributes from either dict or dataclass."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
        if obj is None:
            return 0.0
    return obj


def compute_layer_summary(layer_result) -> LayerSummary:
    """
    Compute summary metrics from a LayerAnalysisResult.
    """
    routing = layer_result.routing_results
    writing = layer_result.writing_results
    programs = layer_result.program_results
    rope = layer_result.rope_stability
    
    n_heads = len(routing)
    
    # Routing averages - handle both dict and dataclass
    avg_diagonal = sum(_get(r, "metrics", "diagonal_dominance") for r in routing) / n_heads
    avg_selectivity = sum(_get(r, "metrics", "top1_mass_mean") for r in routing) / n_heads
    avg_diag_mass = sum(_get(r, "metrics", "diagonal_softmax_mass") for r in routing) / n_heads
    avg_gap = sum(_get(r, "metrics", "max_gap_mean") for r in routing) / n_heads
    avg_asym = sum(_get(r, "metrics", "asymmetry_score") for r in routing) / n_heads
    avg_rank = sum(_get(r, "metrics", "effective_rank") for r in routing) / n_heads
    
    # Head-level dispersion (std across heads)
    selectivity_values = [_get(r, "metrics", "top1_mass_mean") for r in routing]
    diag_mass_values = [_get(r, "metrics", "diagonal_softmax_mass") for r in routing]
    
    import statistics
    std_selectivity = statistics.stdev(selectivity_values) if len(selectivity_values) > 1 else 0.0
    std_diag_mass = statistics.stdev(diag_mass_values) if len(diag_mass_values) > 1 else 0.0
    
    # Baseline metrics (use CORRECT baseline for each metric!)
    # - Selectivity baseline: random_weights (changes logit distribution, NOT column-invariant)
    # - Diagonal identity baseline: permuted-K or indep bases (kills feature identity)
    baseline_rand_sel = 0.0
    baseline_permk_diag_mass = 0.0
    baseline_indep_diag = 0.0
    
    for r in routing:
        # Random weights baseline for selectivity (correct: not permutation-invariant)
        rand_w = _get(r, "baseline_random_weights")
        if rand_w:
            baseline_rand_sel += _get(rand_w, "top1_mass_mean")
        
        # Permuted-K baseline for diagonal mass (correct: kills feature identity)
        permk = _get(r, "baseline_permuted_k")
        if permk:
            baseline_permk_diag_mass += _get(permk, "diagonal_softmax_mass")
        
        # Independent bases baseline for diagonal dominance
        indep = _get(r, "baseline_independent_bases")
        if indep:
            baseline_indep_diag += _get(indep, "diagonal_dominance")
    
    baseline_rand_sel /= n_heads
    baseline_permk_diag_mass /= n_heads
    baseline_indep_diag /= n_heads
    
    # Writing averages
    avg_copy = sum(_get(w, "metrics", "copy_score") for w in writing) / n_heads
    avg_transform = sum(_get(w, "metrics", "transform_score") for w in writing) / n_heads
    avg_diversity = sum(_get(w, "metrics", "write_cosine_diversity") for w in writing) / n_heads
    
    # Median of top write similarities (diagnoses SUPPRESS overconfidence)
    # If median is near 0, top "matches" are weak and classification is suspect
    top_write_sims = []
    for w in writing:
        top_pairs = _get(w, "metrics", "top_transform_pairs")
        if top_pairs and len(top_pairs) > 0:
            top_write_sims.append(abs(top_pairs[0][2]))  # abs of top similarity
    median_top_write_sim = statistics.median(top_write_sims) if top_write_sims else 0.0
    
    # RoPE stability
    avg_ctrl = sum(_get(r, "semantic_controllability") for r in rope) / n_heads
    
    # Program distribution
    all_counts = defaultdict(int)
    for p in programs:
        prog_counts = _get(p, "program_counts")
        if isinstance(prog_counts, dict):
            for ptype, count in prog_counts.items():
                all_counts[ptype] += count
    
    total_programs = sum(all_counts.values()) or 1
    probs = {k: v / total_programs for k, v in all_counts.items()}
    
    dominant_type = max(all_counts.keys(), key=lambda k: all_counts[k]) if all_counts else "unknown"
    
    import math
    entropy = -sum(p * math.log(p + 1e-10) for p in probs.values())
    
    # Cross-head redundancy (average Jaccard)
    redundancy_vals = list(layer_result.cross_head_redundancy.values())
    avg_redundancy = sum(redundancy_vals) / len(redundancy_vals) if redundancy_vals else 0.0
    
    # Get SAE layer used (default to layer_idx if not present)
    sae_layer = getattr(layer_result, 'sae_layer_used', layer_result.layer_idx)
    
    return LayerSummary(
        layer_idx=layer_result.layer_idx,
        is_sliding_window=layer_result.is_sliding_window,
        sae_layer_used=sae_layer,
        avg_diagonal_dominance=avg_diagonal,
        avg_selectivity=avg_selectivity,
        avg_diagonal_softmax_mass=avg_diag_mass,
        avg_max_gap=avg_gap,
        avg_asymmetry=avg_asym,
        avg_effective_rank=avg_rank,
        std_selectivity=std_selectivity,
        std_diagonal_softmax_mass=std_diag_mass,
        avg_copy_score=avg_copy,
        avg_transform_score=avg_transform,
        avg_write_diversity=avg_diversity,
        median_top_write_similarity=median_top_write_sim,
        avg_semantic_controllability=avg_ctrl,
        dominant_program_type=dominant_type,
        program_type_entropy=entropy,
        cross_head_redundancy=avg_redundancy,
        baseline_rand_weights_selectivity=baseline_rand_sel,
        baseline_permuted_k_diag_mass=baseline_permk_diag_mass,
        baseline_indep_bases_diagdom=baseline_indep_diag,
    )


def identify_depth_patterns(summaries: List[LayerSummary]) -> DepthPattern:
    """
    Identify patterns across depth and recommend steering layers.
    """
    if not summaries:
        return DepthPattern(
            high_controllability_layers=[], high_selectivity_layers=[],
            low_redundancy_layers=[], steering_band_start=0, steering_band_end=0,
            steering_band_score=0.0, controllability_trend="flat",
            selectivity_trend="flat", copy_vs_transform_trend="flat"
        )
    
    summaries = sorted(summaries, key=lambda x: x.layer_idx)
    layers = [s.layer_idx for s in summaries]
    
    # Identify high-value layers
    ctrl_threshold = sum(s.avg_semantic_controllability for s in summaries) / len(summaries)
    select_threshold = sum(s.avg_selectivity for s in summaries) / len(summaries)
    redund_threshold = sum(s.cross_head_redundancy for s in summaries) / len(summaries)
    
    high_ctrl = [s.layer_idx for s in summaries if s.avg_semantic_controllability > ctrl_threshold]
    high_select = [s.layer_idx for s in summaries if s.avg_selectivity > select_threshold]
    low_redund = [s.layer_idx for s in summaries if s.cross_head_redundancy < redund_threshold]
    
    # Steering score: high controllability + high selectivity + low redundancy
    scores = {}
    for s in summaries:
        score = (
            s.avg_semantic_controllability * 0.4 +
            s.avg_selectivity * 0.3 +
            (1 - s.cross_head_redundancy) * 0.2 +
            s.avg_max_gap * 0.1
        )
        scores[s.layer_idx] = score
    
    # Find best contiguous band
    best_start, best_end, best_score = 0, 0, 0
    for i, s1 in enumerate(summaries):
        for j in range(i, len(summaries)):
            band_score = sum(scores[summaries[k].layer_idx] for k in range(i, j + 1)) / (j - i + 1)
            if band_score > best_score:
                best_start = summaries[i].layer_idx
                best_end = summaries[j].layer_idx
                best_score = band_score
    
    # Detect trends
    def detect_trend(values: List[float]) -> str:
        if len(values) < 3:
            return "flat"
        
        first_third = sum(values[:len(values)//3]) / (len(values)//3 or 1)
        mid_third_start = len(values)//3
        mid_third_end = 2 * len(values)//3
        mid_third = sum(values[mid_third_start:mid_third_end]) / (mid_third_end - mid_third_start or 1)
        last_third = sum(values[2*len(values)//3:]) / (len(values) - 2*len(values)//3 or 1)
        
        if mid_third > first_third and mid_third > last_third:
            return "peaked"
        elif first_third < mid_third < last_third:
            return "increasing"
        elif first_third > mid_third > last_third:
            return "decreasing"
        return "flat"
    
    ctrl_values = [s.avg_semantic_controllability for s in summaries]
    select_values = [s.avg_selectivity for s in summaries]
    copy_vs_trans = [s.avg_copy_score - s.avg_transform_score for s in summaries]
    
    return DepthPattern(
        high_controllability_layers=high_ctrl,
        high_selectivity_layers=high_select,
        low_redundancy_layers=low_redund,
        steering_band_start=best_start,
        steering_band_end=best_end,
        steering_band_score=best_score,
        controllability_trend=detect_trend(ctrl_values),
        selectivity_trend=detect_trend(select_values),
        copy_vs_transform_trend=detect_trend(copy_vs_trans),
    )


def generate_synthesis_report(run) -> str:
    """
    Generate a human-readable synthesis report with:
    - Uniform baseline comparisons (xUniform metrics)
    - Baseline structure comparisons (permuted-K, independent bases)
    - SAE layer tracking
    - Improved formatting for small values
    """
    lines = []
    lines.append("=" * 80)
    lines.append("WEIGHT-SPACE SAE ANALYSIS SYNTHESIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Run ID: {run.run_id}")
    lines.append(f"Timestamp: {run.timestamp}")
    lines.append(f"Layers analyzed: {run.layers_analyzed}")
    
    # Get n_features for uniform baseline calculation
    n_features = run.config.get("feature_subset_size", 2048)
    uniform_top1 = 1.0 / n_features
    sae_offset = run.config.get("sae_layer_offset_for_attn", 0)
    
    lines.append("")
    lines.append("UNIFORM BASELINE REFERENCE")
    lines.append("-" * 80)
    lines.append(f"n_features (subset size): {n_features}")
    lines.append(f"uniform_top1 = 1/n = {uniform_top1:.6f}")
    lines.append(f"SAE layer offset for attention: {sae_offset}")
    lines.append("Sel x U = selectivity / uniform_top1 (1.0 = uniform, higher = more selective)")
    lines.append("")
    
    # Compute summaries
    summaries = []
    for layer_idx, result in run.layer_results.items():
        summary = compute_layer_summary(result)
        summaries.append(summary)
    
    summaries = sorted(summaries, key=lambda x: x.layer_idx)
    
    # Check if baselines are available
    has_sel_baseline = any(s.baseline_rand_weights_selectivity > 0 for s in summaries)
    has_diag_baseline = any(s.baseline_permuted_k_diag_mass > 0 or s.baseline_indep_bases_diagdom > 0 for s in summaries)
    
    # ================== TABLE 1: Layer Summaries (scaled) ==================
    lines.append("LAYER SUMMARIES (scaled for interpretability)")
    lines.append("-" * 80)
    lines.append(f"{'Layer':>5} {'SAE':>4} {'Slide':>5} {'Sel_xU':>7} {'DiagM_xU':>9} {'MaxGap':>7} {'Ctrl':>6} {'Copy':>6} {'Trans':>6}")
    lines.append("-" * 80)
    
    for s in summaries:
        sw = "Yes" if s.is_sliding_window else "No"
        sel_xu = s.avg_selectivity / uniform_top1
        diag_mass_xu = s.avg_diagonal_softmax_mass / uniform_top1
        lines.append(
            f"{s.layer_idx:>5} {s.sae_layer_used:>4} {sw:>5} "
            f"{sel_xu:>7.2f} {diag_mass_xu:>9.2f} {s.avg_max_gap:>7.2f} "
            f"{s.avg_semantic_controllability:>6.3f} "
            f"{s.avg_copy_score:>6.3f} {s.avg_transform_score:>6.3f}"
        )
    
    lines.append("")
    lines.append("Sel_xU = top1 softmax mass / uniform (permutation-invariant peakiness)")
    lines.append("DiagM_xU = self-probability mass / uniform (identity-sensitive!)")
    lines.append("")
    
    # ================== TABLE 2: Baseline Comparisons ==================
    if has_sel_baseline or has_diag_baseline:
        lines.append("STRUCTURE VS BASELINES (with ratios and dispersion)")
        lines.append("-" * 90)
        lines.append(f"{'Layer':>5} {'Sel_xU':>7} {'RndW':>6} {'Sel/RndW':>8} {'Sel_std':>7} {'DiagM':>6} {'PermK':>6} {'DiagM/P':>7} {'DiagM_std':>9}")
        lines.append("-" * 90)
        
        for s in summaries:
            sel_xu = s.avg_selectivity / uniform_top1
            rand_sel_xu = s.baseline_rand_weights_selectivity / uniform_top1
            sel_ratio = sel_xu / max(rand_sel_xu, 0.01)  # Sel / RndW ratio
            sel_std_xu = s.std_selectivity / uniform_top1
            
            diag_mass_xu = s.avg_diagonal_softmax_mass / uniform_top1
            permk_diag_mass_xu = s.baseline_permuted_k_diag_mass / uniform_top1
            diag_ratio = diag_mass_xu / max(permk_diag_mass_xu, 0.01)
            diag_std_xu = s.std_diagonal_softmax_mass / uniform_top1
            
            lines.append(
                f"{s.layer_idx:>5} {sel_xu:>7.2f} {rand_sel_xu:>6.2f} {sel_ratio:>8.2f}x {sel_std_xu:>7.2f} "
                f"{diag_mass_xu:>6.2f} {permk_diag_mass_xu:>6.2f} {diag_ratio:>7.2f}x {diag_std_xu:>9.2f}"
            )
        
        lines.append("")
        lines.append("Sel/RndW: selectivity as multiple of random-weights baseline (>1 = real structure)")
        lines.append("DiagM/P: diagonal mass as multiple of permuted-K baseline (>1 = identity signal)")
        lines.append("_std columns show dispersion across heads (high = few heads carry structure)")
        lines.append("")
    
    # ================== Write Quality Diagnostics ==================
    lines.append("WRITE QUALITY DIAGNOSTICS")
    lines.append("-" * 80)
    lines.append(f"{'Layer':>5} {'MedTopWriteSim':>14} {'Interpretation':>40}")
    lines.append("-" * 80)
    
    for s in summaries:
        med_sim = s.median_top_write_similarity
        if med_sim < 0.05:
            interp = "WEAK: top matches near 0, SUPPRESS suspect"
        elif med_sim < 0.15:
            interp = "MODERATE: some match strength"
        else:
            interp = "STRONG: clear write targets"
        lines.append(f"{s.layer_idx:>5} {med_sim:>14.3f} {interp:>40}")
    
    lines.append("")
    
    # ================== Depth Patterns ==================
    patterns = identify_depth_patterns(summaries)
    
    lines.append("DEPTH PATTERNS")
    lines.append("-" * 80)
    lines.append(f"High controllability layers: {patterns.high_controllability_layers}")
    lines.append(f"High selectivity layers: {patterns.high_selectivity_layers}")
    lines.append(f"Low redundancy layers: {patterns.low_redundancy_layers}")
    lines.append("")
    lines.append(f"Controllability trend: {patterns.controllability_trend}")
    lines.append(f"Selectivity trend: {patterns.selectivity_trend}")
    lines.append(f"Copy vs Transform trend: {patterns.copy_vs_transform_trend}")
    lines.append("")
    
    # ================== Steering Recommendation ==================
    lines.append("STEERING RECOMMENDATION")
    lines.append("-" * 80)
    lines.append(f"Recommended steering band: layers {patterns.steering_band_start} - {patterns.steering_band_end}")
    lines.append(f"Steering score: {patterns.steering_band_score:.4f}")
    
    # Compute baseline-corrected steering if baselines available
    if has_sel_baseline or has_diag_baseline:
        # Recompute scores using selectivity uplift instead of raw selectivity
        corrected_scores = {}
        for s in summaries:
            sel_signal = max(s.avg_selectivity - s.baseline_rand_weights_selectivity, 0)
            diag_signal = max(s.avg_diagonal_softmax_mass - s.baseline_permuted_k_diag_mass, 0)
            # Normalize by uniform
            sel_norm = sel_signal / uniform_top1
            diag_norm = diag_signal / uniform_top1
            # Weighted score
            score = (
                s.avg_semantic_controllability * 0.3 +
                (sel_norm / (n_features / 10)) * 0.3 +  # Normalize to reasonable scale
                (diag_norm / (n_features / 10)) * 0.2 +
                (1 - s.cross_head_redundancy) * 0.2
            )
            corrected_scores[s.layer_idx] = score
        
        if corrected_scores:
            best_corrected = max(corrected_scores.items(), key=lambda x: x[1])
            lines.append(f"Baseline-corrected best layer: {best_corrected[0]} (score: {best_corrected[1]:.4f})")
    
    lines.append("")
    
    # ================== Interpretation ==================
    lines.append("INTERPRETATION")
    lines.append("-" * 80)
    
    # Metric explanations
    lines.append("* Sel_xU = top-1 softmax mass / uniform. Measures row peakiness.")
    lines.append(f"  (uniform = 1/n = {uniform_top1:.6f}, so Sel_xU=1 means uniform)")
    lines.append("")
    lines.append("* DiagM_xU = diagonal softmax mass / uniform. Measures self-attention.")
    lines.append("  (This is IDENTITY-SENSITIVE! Unlike Sel_xU which is permutation-invariant)")
    lines.append("")
    
    if has_sel_baseline or has_diag_baseline:
        lines.append("* Sel/RndW compares to RANDOM-WEIGHTS baseline (temperature-matched):")
        lines.append("  >1 = heads have structure beyond random projection noise.")
        lines.append("  ~1 = no more selective than random; <1 = flatter than random (rare).")
        lines.append("")
        lines.append("* DiagM/P compares to PERMUTED-K baseline (kills feature identity):")
        lines.append("  >1 = real feature-identity self-match beyond marginal distribution.")
        lines.append("  ~1 = diagonal mass is just chance alignment.")
        lines.append("")
    
    if patterns.controllability_trend == "peaked":
        lines.append("* Controllability peaks in middle layers - classic 'steering sweet spot' pattern.")
    elif patterns.controllability_trend == "increasing":
        lines.append("* Controllability increases with depth - later layers more semantically controlled.")
    elif patterns.controllability_trend == "decreasing":
        lines.append("* Controllability decreases with depth - early layers more stable to position.")
    
    if patterns.selectivity_trend == "increasing":
        lines.append("* Selectivity increases with depth - heads become more specialized.")
    elif patterns.selectivity_trend == "peaked":
        lines.append("* Selectivity peaks in middle layers - balanced specialization.")
    
    lines.append("")
    
    # ================== Program Distribution ==================
    # Track both explicit-write programs and fallback programs separately
    program_counts_all = defaultdict(int)
    program_counts_explicit = defaultdict(int)
    fallback_by_type = defaultdict(int)
    total_fallback = 0
    
    for layer_idx, result in run.layer_results.items():
        for p in result.program_results:
            # Get program list with fallback info
            top_programs = _get(p, "top_programs") or []
            fallback_count = _get(p, "fallback_reinforce_count") or 0
            total_fallback += fallback_count
            
            # Also get counts from program_counts dict
            prog_counts = _get(p, "program_counts")
            if isinstance(prog_counts, dict):
                for ptype, count in prog_counts.items():
                    ptype_str = str(ptype).replace("ProgramType.", "") if "ProgramType" in str(ptype) else str(ptype)
                    program_counts_all[ptype_str] += count
            
            # Analyze individual programs for explicit vs fallback
            for prog in top_programs:
                if isinstance(prog, dict):
                    ptype = prog.get("program_type", "unknown")
                    is_fallback = prog.get("used_fallback_write", False)
                else:
                    ptype = getattr(prog, "program_type", "unknown")
                    is_fallback = getattr(prog, "used_fallback_write", False)
                
                ptype_str = str(ptype).replace("ProgramType.", "").replace("reinforce", "REINFORCE").replace("cross_copy", "CROSS_COPY").replace("shift", "SHIFT").replace("transform", "TRANSFORM").replace("relay", "RELAY").replace("suppress", "SUPPRESS")
                
                if not is_fallback:
                    program_counts_explicit[ptype_str] += 1
                else:
                    fallback_by_type[ptype_str] += 1
    
    # Print EXPLICIT-ONLY histogram (trustworthy)
    lines.append("PROGRAM DISTRIBUTION - EXPLICIT WRITES ONLY (trustworthy)")
    lines.append("-" * 80)
    lines.append("(These programs have real W2F evidence, not synthetic self-write injection)")
    total_explicit = sum(program_counts_explicit.values())
    if total_explicit > 0:
        for ptype, count in sorted(program_counts_explicit.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total_explicit if total_explicit > 0 else 0
            lines.append(f"  {ptype:15}: {count:5} ({pct:5.1f}%)")
    else:
        lines.append("  (No explicit-write programs found)")
    lines.append(f"  Total explicit: {total_explicit}")
    lines.append("")
    
    # Print ALL programs histogram (for comparison)
    lines.append("PROGRAM DISTRIBUTION - INCLUDING FALLBACK (may be inflated)")
    lines.append("-" * 80)
    lines.append("(Includes synthetic self-writes from copy_score injection)")
    total_all = sum(program_counts_all.values())
    if total_all > 0:
        for ptype, count in sorted(program_counts_all.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total_all if total_all > 0 else 0
            # Show fallback count for this type if any
            fb = fallback_by_type.get(ptype, 0)
            fb_marker = f" (incl. {fb} fallback)" if fb > 0 else ""
            lines.append(f"  {ptype:15}: {count:5} ({pct:5.1f}%){fb_marker}")
    else:
        lines.append("  (No programs found)")
    lines.append(f"  Total (all): {total_all}, Total fallback: {total_fallback}")
    lines.append("")
    
    # Warning if lots of fallback
    if total_fallback > 0 and total_all > 0:
        fallback_pct = 100 * total_fallback / total_all
        if fallback_pct > 20:
            lines.append(f"⚠ WARNING: {fallback_pct:.1f}% of programs used fallback self-write injection.")
            lines.append("  The 'explicit-only' histogram is more trustworthy for circuit analysis.")
            lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Synthesis module loaded.")
    print("Use generate_synthesis_report(run) to generate summary.")
