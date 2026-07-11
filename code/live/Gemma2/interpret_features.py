#!/usr/bin/env python3
"""
interpret_features.py - Semantic interpretation of weight-space analysis results.

Reads analysis JSON files and queries Neuronpedia to produce human-readable
interpretation of:
  - Top K routing pairs (i→j) with scores and feature descriptions
  - Top K write pairs (j→k) with scores and feature descriptions  
  - Top K composed programs (i→j→k) if available
  
Usage:
  python interpret_features.py analysis_outputs/analysis_20260102_025426.json --layer 6 --head 3
  python interpret_features.py analysis_outputs/analysis_20260102_025426.json --layer 6 --all-heads
  python interpret_features.py analysis_outputs/analysis_20260102_025426.json --top-heads 5
"""

import os
import sys
import json
import argparse
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from getpass import getpass
from datetime import datetime

# Optional: Neuronpedia integration
try:
    from neuronpedia.np_sae_feature import SAEFeature
    HAS_NEURONPEDIA = True
except ImportError:
    HAS_NEURONPEDIA = False
    print("Warning: neuronpedia not installed. Feature descriptions will be unavailable.")


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH = "16k"  # For Neuronpedia source string

TOP_K_ROUTING = 10
TOP_K_WRITING = 10
TOP_K_PROGRAMS = 10

NP_CACHE_PATH = "neuronpedia_cache.json"

# =============================================================================
# Neuronpedia API Key (set this to skip interactive prompt)
# Get your key from: https://www.neuronpedia.org/api-key
# =============================================================================
NEURONPEDIA_API_KEY = "sk-np-VRTZse3U0xgVmJmW0qoIQnx6qPecUPKS6KLVRwsQksk0"  # Set to your key string, e.g., "np-xxxxxxxxxxxxxxxx"


# =============================================================================
# Neuronpedia helpers (adapted from gemma2_initial.py)
# =============================================================================

_np_disk_cache = {}

def load_np_cache():
    global _np_disk_cache
    if os.path.exists(NP_CACHE_PATH):
        try:
            with open(NP_CACHE_PATH, "r") as f:
                _np_disk_cache = json.load(f)
            print(f"Loaded Neuronpedia cache: {len(_np_disk_cache)} entries")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            _np_disk_cache = {}

def save_np_cache():
    try:
        with open(NP_CACHE_PATH, "w") as f:
            json.dump(_np_disk_cache, f)
        print(f"Saved Neuronpedia cache: {len(_np_disk_cache)} entries")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def _np_key(sae_layer: int, feature_idx: int) -> str:
    return f"{MODEL_ID}|{sae_layer}-gemmascope-res-{SAE_WIDTH}|{feature_idx}"


@functools.lru_cache(maxsize=8192)
def get_feature_info(sae_layer: int, feature_idx: int) -> Dict:
    """
    Get feature info from Neuronpedia. Returns dict with:
      - description: short description or top tokens
      - top_tokens: list of top activating tokens
      - snippets: list of example activation contexts (like gemma2_initial.py)
    """
    key = _np_key(sae_layer, feature_idx)
    
    # Check cache first
    if key in _np_disk_cache:
        return _np_disk_cache[key]
    
    if not HAS_NEURONPEDIA or not os.getenv("NEURONPEDIA_API_KEY"):
        return {"description": f"[Feature {feature_idx}]", "top_tokens": [], "snippets": []}
    
    try:
        source = f"{sae_layer}-gemmascope-res-{SAE_WIDTH}"
        f = SAEFeature.get(MODEL_ID, source, str(int(feature_idx)))
        
        # Extract json data
        jd = getattr(f, "jsonData", None) or getattr(f, "json_data", None)
        if isinstance(jd, str):
            try:
                payload = json.loads(jd)
            except:
                payload = {}
        elif isinstance(jd, dict):
            payload = jd
        else:
            payload = getattr(f, "__dict__", {}) or {}
        
        # Extract useful fields - get ALL tokens, no limits
        pos_str = payload.get("pos_str", [])  # All positive tokens
        neg_str = payload.get("neg_str", [])  # All negative tokens
        pos_values = payload.get("pos_values", [])
        neg_values = payload.get("neg_values", [])
        
        # Get ALL example snippets from Neuronpedia - no trimming for interpretability
        snippets = []
        seen_snippets = set()  # For de-duplication only
        
        for ex in (payload.get("activations") or []):  # Get ALL activations
            # Get tokens - these are the actual text tokens from the example
            toks = ex.get("token") or ex.get("tokens") or []
            if not toks:
                continue
            
            # Join tokens to form readable text - NO TRIMMING
            txt_parts = []
            for t in toks:
                t_str = str(t)
                # Clean up special tokens
                if t_str.startswith("<") and t_str.endswith(">"):
                    continue  # Skip special tokens like <bos>
                txt_parts.append(t_str)
            txt = "".join(txt_parts).replace("▁", " ").replace("\n", " ").strip()
            
            if not txt:
                continue
            
            # De-duplicate by checking similarity (first 50 chars)
            snippet_key = txt[:50].lower()
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
            
            snippets.append(txt)  # Full text, no trimming
        
        # Build description from ALL top tokens with values if available
        if pos_str:
            top_with_vals = []
            for i, tok in enumerate(pos_str):  # ALL tokens, no limit
                if i < len(pos_values):
                    top_with_vals.append(f"'{tok}'({pos_values[i]:.1f})")
                else:
                    top_with_vals.append(f"'{tok}'")
            desc = "top: " + ", ".join(top_with_vals)
        else:
            desc = f"[Feature {feature_idx}]"
        
        result = {
            "description": desc,
            "top_tokens": pos_str,
            "neg_tokens": neg_str,
            "snippets": snippets,
        }
        
        _np_disk_cache[key] = result
        return result
        
    except Exception as e:
        result = {"description": f"[Feature {feature_idx}] (error: {str(e)[:30]})", "top_tokens": [], "snippets": []}
        return result


def format_feature(sae_layer: int, feature_idx: int, verbose: bool = True) -> str:
    """
    Format a feature with its Neuronpedia description and example snippets.
    
    Args:
        sae_layer: SAE layer index
        feature_idx: Feature index within SAE
        verbose: If True, include ALL example snippets from Neuronpedia
    """
    info = get_feature_info(sae_layer, feature_idx)
    desc = info["description"]
    lines = [f"F{feature_idx}: {desc}"]
    
    # Add ALL snippets if available and verbose mode - no limits for interpretability
    if verbose and info.get("snippets"):
        for i, snip in enumerate(info["snippets"]):  # Show ALL snippets
            lines.append(f"         ex{i+1}: \"{snip}\"")
    
    return "\n".join(lines)


# =============================================================================
# Analysis data extraction
# =============================================================================

def load_analysis(json_path: str) -> Dict:
    """Load analysis JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def get_layer_data(analysis: Dict, layer_idx: int) -> Optional[Dict]:
    """Get data for a specific layer."""
    layer_results = analysis.get("layer_results", {})
    return layer_results.get(str(layer_idx))


def get_sae_layer(layer_idx: int, analysis: Dict) -> int:
    """Get the SAE layer used for this attention layer."""
    config = analysis.get("config", {})
    offset = config.get("sae_layer_offset_for_attn", -1)
    return max(0, layer_idx + offset)


def extract_routing_pairs(layer_data: Dict, head_idx: int) -> List[Tuple[int, int, float]]:
    """Extract top routing pairs for a head."""
    routing = layer_data.get("routing_results", [])
    if head_idx >= len(routing):
        return []
    
    head_routing = routing[head_idx]
    top_pairs = head_routing.get("top_pairs", {})
    
    # Handle both dict and list formats
    if isinstance(top_pairs, dict):
        positive = top_pairs.get("positive_pairs", [])
        negative = top_pairs.get("negative_pairs", [])
        # Combine and sort by absolute score
        all_pairs = [(p[0], p[1], p[2]) for p in positive] + [(p[0], p[1], p[2]) for p in negative]
        all_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return all_pairs
    elif isinstance(top_pairs, list):
        return [(p[0], p[1], p[2]) for p in top_pairs]
    return []


def extract_writing_pairs(layer_data: Dict, head_idx: int) -> List[Tuple[int, int, float]]:
    """Extract top write pairs for a head."""
    writing = layer_data.get("writing_results", [])
    if head_idx >= len(writing):
        return []
    
    head_writing = writing[head_idx]
    metrics = head_writing.get("metrics", {})
    
    # Try different field names for write pairs
    pairs = (metrics.get("top_transform_pairs") or 
             metrics.get("top_write_pairs") or 
             metrics.get("top_copy_pairs") or [])
    
    return [(p[0], p[1], p[2]) for p in pairs[:TOP_K_WRITING]]


def extract_programs(layer_data: Dict, head_idx: int) -> Dict:
    """Extract program information for a head."""
    programs = layer_data.get("program_results", [])
    if head_idx >= len(programs):
        return {}
    
    head_programs = programs[head_idx]
    return {
        "program_counts": head_programs.get("program_counts", {}),
        "top_programs": head_programs.get("top_programs", []),  # Full program list
        "top_sources": head_programs.get("top_sources", []),
        "broadcast_sources": head_programs.get("broadcast_sources", []),
        "fallback_reinforce_count": head_programs.get("fallback_reinforce_count", 0),
    }


def get_head_metrics(layer_data: Dict, head_idx: int) -> Dict:
    """Get key metrics for a head."""
    routing = layer_data.get("routing_results", [])
    writing = layer_data.get("writing_results", [])
    rope = layer_data.get("rope_stability", [])
    
    metrics = {}
    
    if head_idx < len(routing):
        r = routing[head_idx].get("metrics", {})
        metrics["selectivity"] = r.get("top1_mass_mean", 0)
        metrics["diagonal_dominance"] = r.get("diagonal_dominance", 0)
        metrics["diagonal_softmax_mass"] = r.get("diagonal_softmax_mass", 0)
        metrics["max_gap"] = r.get("max_gap_mean", 0)
        metrics["asymmetry"] = r.get("asymmetry_score", 0)
        metrics["archetype"] = routing[head_idx].get("archetype", "unknown")
    
    if head_idx < len(writing):
        w = writing[head_idx].get("metrics", {})
        metrics["copy_score"] = w.get("copy_score", 0)
        metrics["transform_score"] = w.get("transform_score", 0)
        metrics["write_archetype"] = writing[head_idx].get("archetype", "unknown")
    
    if head_idx < len(rope):
        metrics["controllability"] = rope[head_idx].get("semantic_controllability", 0)
    
    return metrics


# =============================================================================
# Report generation
# =============================================================================

def generate_head_report(
    analysis: Dict,
    layer_idx: int,
    head_idx: int,
    use_neuronpedia: bool = True,
) -> str:
    """Generate interpretable report for a single head."""
    
    layer_data = get_layer_data(analysis, layer_idx)
    if not layer_data:
        return f"Layer {layer_idx} not found in analysis."
    
    sae_layer = get_sae_layer(layer_idx, analysis)
    config = analysis.get("config", {})
    n_features = config.get("feature_subset_size", 2048)
    uniform = 1.0 / n_features
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"FEATURE INTERPRETATION: Layer {layer_idx}, Head {head_idx}")
    lines.append(f"SAE Layer Used: {sae_layer}")
    lines.append("=" * 80)
    lines.append("")
    
    # Metrics summary
    metrics = get_head_metrics(layer_data, head_idx)
    lines.append("HEAD METRICS")
    lines.append("-" * 40)
    sel_xu = metrics.get("selectivity", 0) / uniform
    diag_xu = metrics.get("diagonal_softmax_mass", 0) / uniform
    lines.append(f"  Selectivity (Sel_xU):     {sel_xu:.2f}x uniform")
    lines.append(f"  Diagonal Mass (DiagM_xU): {diag_xu:.2f}x uniform")
    lines.append(f"  Max Gap:                  {metrics.get('max_gap', 0):.3f}")
    lines.append(f"  Controllability:          {metrics.get('controllability', 0):.3f}")
    lines.append(f"  Copy Score:               {metrics.get('copy_score', 0):.3f}")
    lines.append(f"  Transform Score:          {metrics.get('transform_score', 0):.3f}")
    lines.append(f"  Routing Archetype:        {metrics.get('archetype', 'unknown')}")
    lines.append(f"  Writing Archetype:        {metrics.get('write_archetype', 'unknown')}")
    lines.append("")
    
    # Top routing pairs
    lines.append("TOP ROUTING PAIRS (query feature → attends to → key feature)")
    lines.append("-" * 80)
    routing_pairs = extract_routing_pairs(layer_data, head_idx)[:TOP_K_ROUTING]
    
    if routing_pairs:
        for i, (q_idx, k_idx, score) in enumerate(routing_pairs, 1):
            sign = "+" if score > 0 else "-"
            is_self_loop = (q_idx == k_idx)
            loop_marker = " ← SELF-LOOP" if is_self_loop else ""
            lines.append(f"{i:2}. [{sign}{abs(score):.3f}] {q_idx} → {k_idx}{loop_marker}")
            if use_neuronpedia:
                q_info = format_feature(sae_layer, q_idx, verbose=True)
                lines.append(f"     Query: {q_info}")
                if not is_self_loop:
                    k_info = format_feature(sae_layer, k_idx, verbose=True)
                    lines.append(f"     Key:   {k_info}")
            lines.append("")
    else:
        lines.append("  No routing pairs found.")
    lines.append("")
    
    # TOP OFF-DIAGONAL ROUTING (for SHIFT/RELAY/CROSS_COPY evidence)
    all_routing = extract_routing_pairs(layer_data, head_idx)
    off_diagonal = [(q, k, s) for q, k, s in all_routing if q != k][:10]
    
    if off_diagonal:
        lines.append("TOP OFF-DIAGONAL ROUTING (cross-feature attention)")
        lines.append("-" * 80)
        for i, (q_idx, k_idx, score) in enumerate(off_diagonal, 1):
            sign = "+" if score > 0 else "-"
            lines.append(f"{i:2}. [{sign}{abs(score):.3f}] {q_idx} → {k_idx}")
            if use_neuronpedia:
                q_info = format_feature(sae_layer, q_idx, verbose=True)
                k_info = format_feature(sae_layer, k_idx, verbose=True)
                lines.append(f"     Query: {q_info}")
                lines.append(f"     Key:   {k_info}")
            lines.append("")
        lines.append("")
    
    # Top write pairs
    lines.append("TOP WRITE PAIRS (attended feature → writes to → output feature)")
    lines.append("-" * 80)
    writing_pairs = extract_writing_pairs(layer_data, head_idx)[:TOP_K_WRITING]
    
    if writing_pairs:
        for i, (j_idx, k_idx, score) in enumerate(writing_pairs, 1):
            sign = "+" if score > 0 else "-"
            lines.append(f"{i:2}. [{sign}{abs(score):.3f}] {j_idx} → {k_idx}")
            if use_neuronpedia:
                j_info = format_feature(sae_layer, j_idx, verbose=True)
                k_info = format_feature(sae_layer, k_idx, verbose=True)
                lines.append(f"     Input:  {j_info}")
                lines.append(f"     Output: {k_info}")
            lines.append("")
    else:
        lines.append("  No write pairs found.")
    lines.append("")
    
    # Program distribution - split by explicit vs fallback
    program_info = extract_programs(layer_data, head_idx)
    prog_counts = program_info.get("program_counts", {})
    top_programs = program_info.get("top_programs", [])
    fallback_count = program_info.get("fallback_reinforce_count", 0)
    
    # Count explicit vs fallback from individual programs
    from collections import defaultdict
    explicit_by_type = defaultdict(int)
    fallback_by_type = defaultdict(int)
    
    for prog in top_programs:
        if isinstance(prog, dict):
            ptype = prog.get("program_type", "unknown")
            is_fallback = prog.get("used_fallback_write", False)
        else:
            ptype = getattr(prog, "program_type", "unknown")
            is_fallback = getattr(prog, "used_fallback_write", False)
        
        ptype_str = str(ptype).replace("ProgramType.", "").replace("reinforce", "REINFORCE").replace("cross_copy", "CROSS_COPY").replace("shift", "SHIFT").replace("transform", "TRANSFORM").replace("relay", "RELAY").replace("suppress", "SUPPRESS")
        
        if is_fallback:
            fallback_by_type[ptype_str] += 1
        else:
            explicit_by_type[ptype_str] += 1
    
    total_explicit = sum(explicit_by_type.values())
    total_fallback_counted = sum(fallback_by_type.values())
    
    # Print explicit-only (trustworthy)
    if explicit_by_type:
        lines.append("PROGRAM DISTRIBUTION - EXPLICIT WRITES ONLY (trustworthy)")
        lines.append("-" * 60)
        for prog_type, count in sorted(explicit_by_type.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_explicit if total_explicit > 0 else 0
            lines.append(f"  {prog_type:15} {count:4} ({pct:5.1f}%)")
        lines.append(f"  Total explicit: {total_explicit}")
        lines.append("")
    
    # Print all including fallback
    if prog_counts:
        lines.append("PROGRAM DISTRIBUTION - INCLUDING FALLBACK (may be inflated)")
        lines.append("-" * 60)
        total = sum(prog_counts.values())
        for prog_type, count in sorted(prog_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total > 0 else 0
            prog_name = str(prog_type).replace("ProgramType.", "")
            fb = fallback_by_type.get(prog_name.upper(), 0)
            fb_marker = f" (incl. {fb} fallback)" if fb > 0 else ""
            lines.append(f"  {prog_name:15} {count:4} ({pct:5.1f}%){fb_marker}")
        lines.append(f"  Total: {total}, Fallback: {total_fallback_counted}")
        
        # Warning if lots of fallback
        if total_fallback_counted > 0 and total > 0:
            fb_pct = 100 * total_fallback_counted / total
            if fb_pct > 20:
                lines.append(f"  ⚠ {fb_pct:.0f}% are fallback - use explicit-only for analysis")
        lines.append("")
    
    # TOP 5 PROGRAMS PER CATEGORY
    if top_programs:
        lines.append("TOP 5 PROGRAMS PER CATEGORY (i → j → k triplets)")
        lines.append("-" * 80)
        
        # Group programs by type
        from collections import defaultdict
        by_type = defaultdict(list)
        for prog in top_programs:
            # Handle both dict and object formats
            if isinstance(prog, dict):
                ptype = prog.get("program_type", "unknown")
                i = prog.get("query_feature", 0)
                j = prog.get("key_feature", 0)
                k = prog.get("write_feature", 0)
                route = prog.get("route_strength", 0)
                write = prog.get("write_strength", 0)
                score = prog.get("program_score", 0)
                fallback = prog.get("used_fallback_write", False)
            else:
                ptype = getattr(prog, "program_type", "unknown")
                i = getattr(prog, "query_feature", 0)
                j = getattr(prog, "key_feature", 0)
                k = getattr(prog, "write_feature", 0)
                route = getattr(prog, "route_strength", 0)
                write = getattr(prog, "write_strength", 0)
                score = getattr(prog, "program_score", 0)
                fallback = getattr(prog, "used_fallback_write", False)
            
            # Clean up program type
            ptype_clean = str(ptype).replace("ProgramType.", "").replace("reinforce", "REINFORCE").replace("shift", "SHIFT").replace("cross_copy", "CROSS_COPY").replace("transform", "TRANSFORM").replace("relay", "RELAY").replace("suppress", "SUPPRESS")
            by_type[ptype_clean].append((i, j, k, route, write, score, fallback))
        
        # Print top 5 from each category
        for ptype in sorted(by_type.keys()):
            progs = by_type[ptype][:5]  # Top 5
            lines.append(f"\n  {ptype}:")
            for idx, (i, j, k, route, write, score, fallback) in enumerate(progs, 1):
                fallback_mark = " [fallback]" if fallback else ""
                lines.append(f"    {idx}. F{i} → F{j} → F{k} (route={route:.2f}, write={write:.2f}, score={score:.2f}){fallback_mark}")
                if use_neuronpedia:
                    # Show brief descriptions for the triplet
                    i_desc = format_feature(sae_layer, i, verbose=False).split('\n')[0]
                    if j != i:
                        j_desc = format_feature(sae_layer, j, verbose=False).split('\n')[0]
                    else:
                        j_desc = "(same as query)"
                    if k != j and k != i:
                        k_desc = format_feature(sae_layer, k, verbose=False).split('\n')[0]
                    elif k == j:
                        k_desc = "(same as key)"
                    else:
                        k_desc = "(same as query)"
                    lines.append(f"       Query:  {i_desc}")
                    lines.append(f"       Key:    {j_desc}")
                    lines.append(f"       Write:  {k_desc}")
        lines.append("")
    
    # Interpretation hints
    lines.append("INTERPRETATION HINTS")
    lines.append("-" * 40)
    
    archetype = metrics.get("archetype", "").upper()
    if "REPULSION" in archetype:
        lines.append("  ⚠ REPULSION archetype: This head may SUPPRESS certain features.")
        lines.append("    Look at negative routing scores - those are features being pushed away.")
    elif "SELF_MATCH" in archetype:
        lines.append("  ✓ SELF_MATCH archetype: This head reinforces features that attend to themselves.")
        lines.append("    High diagonal mass suggests feature identity preservation.")
    elif "DIFFUSE" in archetype:
        lines.append("  • DIFFUSE archetype: Attention is spread across many features.")
        lines.append("    May be doing aggregation or context mixing.")
    
    if sel_xu > 5:
        lines.append(f"  ✓ HIGH SELECTIVITY ({sel_xu:.1f}x): This head is very picky about what it attends to.")
    elif sel_xu < 1.5:
        lines.append(f"  • LOW SELECTIVITY ({sel_xu:.1f}x): This head spreads attention broadly.")
    
    if diag_xu > 3:
        lines.append(f"  ✓ HIGH SELF-MATCH ({diag_xu:.1f}x): Features strongly attend to themselves.")
    
    if metrics.get("transform_score", 0) > 0.5:
        lines.append(f"  ✓ TRANSFORMER: This head actively changes feature representations.")
    elif metrics.get("copy_score", 0) > 0.3:
        lines.append(f"  ✓ COPIER: This head preserves/copies feature information.")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_layer_summary(
    analysis: Dict,
    layer_idx: int,
    use_neuronpedia: bool = True,
) -> str:
    """Generate summary for all heads in a layer."""
    
    layer_data = get_layer_data(analysis, layer_idx)
    if not layer_data:
        return f"Layer {layer_idx} not found."
    
    n_heads = len(layer_data.get("routing_results", []))
    sae_layer = get_sae_layer(layer_idx, analysis)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"LAYER {layer_idx} SUMMARY (SAE Layer {sae_layer})")
    lines.append("=" * 80)
    lines.append("")
    
    for head_idx in range(n_heads):
        lines.append(generate_head_report(analysis, layer_idx, head_idx, use_neuronpedia))
        lines.append("")
    
    return "\n".join(lines)


def generate_top_heads_report(
    analysis: Dict,
    top_n: int = 5,
    use_neuronpedia: bool = True,
) -> str:
    """Generate report for top N most interesting heads across all layers."""
    
    layer_results = analysis.get("layer_results", {})
    config = analysis.get("config", {})
    n_features = config.get("feature_subset_size", 2048)
    uniform = 1.0 / n_features
    
    # Score all heads
    head_scores = []
    for layer_str, layer_data in layer_results.items():
        layer_idx = int(layer_str)
        routing = layer_data.get("routing_results", [])
        
        for head_idx, head_routing in enumerate(routing):
            metrics = head_routing.get("metrics", {})
            sel = metrics.get("top1_mass_mean", 0)
            diag = metrics.get("diagonal_softmax_mass", 0)
            
            # Simple scoring: selectivity * diagonal mass
            score = (sel / uniform) * (diag / uniform)
            head_scores.append((score, layer_idx, head_idx))
    
    # Sort and take top N
    head_scores.sort(key=lambda x: -x[0])
    top_heads = head_scores[:top_n]
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"TOP {top_n} MOST STRUCTURED HEADS")
    lines.append("=" * 80)
    lines.append("")
    
    for rank, (score, layer_idx, head_idx) in enumerate(top_heads, 1):
        lines.append(f"RANK {rank}: Layer {layer_idx}, Head {head_idx} (structure score: {score:.1f})")
        lines.append("-" * 60)
        lines.append(generate_head_report(analysis, layer_idx, head_idx, use_neuronpedia))
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic interpretation of weight-space analysis results."
    )
    parser.add_argument("json_file", help="Path to analysis JSON file")
    parser.add_argument("--layer", type=int, help="Specific layer to analyze")
    parser.add_argument("--head", type=int, help="Specific head to analyze (requires --layer)")
    parser.add_argument("--all-heads", action="store_true", help="Analyze all heads in layer")
    parser.add_argument("--top-heads", type=int, help="Analyze top N heads across all layers")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--no-neuronpedia", action="store_true", help="Skip Neuronpedia lookups")
    
    args = parser.parse_args()
    
    # Load analysis
    print(f"Loading analysis from: {args.json_file}")
    analysis = load_analysis(args.json_file)
    print(f"Run ID: {analysis.get('run_id', 'unknown')}")
    print(f"Layers: {list(analysis.get('layer_results', {}).keys())}")
    
    # Setup Neuronpedia
    use_neuronpedia = not args.no_neuronpedia
    if use_neuronpedia and HAS_NEURONPEDIA:
        load_np_cache()
        
        # Check for static key first, then env var, then prompt
        if NEURONPEDIA_API_KEY:
            os.environ["NEURONPEDIA_API_KEY"] = NEURONPEDIA_API_KEY
            print("Using static Neuronpedia API key from config.")
        elif not os.getenv("NEURONPEDIA_API_KEY"):
            key = input("Neuronpedia API key (press Enter to skip): ")
            if key.strip():
                os.environ["NEURONPEDIA_API_KEY"] = key.strip()
                print("Neuronpedia key set.")
            else:
                print("Skipping Neuronpedia lookups.")
                use_neuronpedia = False
    
    # Generate report
    if args.top_heads:
        report = generate_top_heads_report(analysis, args.top_heads, use_neuronpedia)
    elif args.layer is not None:
        if args.head is not None:
            report = generate_head_report(analysis, args.layer, args.head, use_neuronpedia)
        elif args.all_heads:
            report = generate_layer_summary(analysis, args.layer, use_neuronpedia)
        else:
            # Default to all heads if layer specified without --head
            report = generate_layer_summary(analysis, args.layer, use_neuronpedia)
    else:
        print("Please specify --layer, --top-heads, or provide other options.")
        print("Examples:")
        print(f"  python {sys.argv[0]} {args.json_file} --layer 6 --head 3")
        print(f"  python {sys.argv[0]} {args.json_file} --layer 6 --all-heads")
        print(f"  python {sys.argv[0]} {args.json_file} --top-heads 5")
        return
    
    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    
    # Save Neuronpedia cache
    if use_neuronpedia and HAS_NEURONPEDIA:
        save_np_cache()


if __name__ == "__main__":
    main()
