"""
Analysis Pipeline Runner for Gemma-2 Weight-Space SAE Analysis.

Main entry point that orchestrates:
- Model and SAE loading
- Per-layer, per-head analysis
- Result aggregation and synthesis
"""

import os
import json
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

from config import Gemma2Config, AnalysisConfig, GEMMA2_CONFIG, ANALYSIS_CONFIG, print_config_summary
from weight_extraction import WeightExtractor, extract_layer_weights
from sae_utils import SAEManager, SAEFeatures, project_to_qk_space
from qk_routing import analyze_head_routing, HeadRoutingResult, compute_cross_head_redundancy
from ov_writing import analyze_head_writing, HeadWriteResult
from feature_programs import analyze_head_programs, HeadProgramResult, summarize_programs
from rope_utils import RoPEAnalyzer


@dataclass
class LayerAnalysisResult:
    """Complete analysis for a single layer."""
    layer_idx: int
    is_sliding_window: bool
    sae_layer_used: int  # SAE layer index used (may differ from layer_idx with offset)
    
    # Per-head results
    routing_results: List[Dict]
    writing_results: List[Dict]
    program_results: List[Dict]
    
    # RoPE stability (per head)
    rope_stability: List[Dict]
    
    # Layer-level aggregates
    cross_head_redundancy: Dict
    head_contribution_ranking: List[int]


@dataclass
class AnalysisRun:
    """Complete analysis run metadata and results."""
    run_id: str
    timestamp: str
    config: Dict
    analysis_config: Dict
    
    layers_analyzed: List[int]
    layer_results: Dict[int, LayerAnalysisResult]
    
    # Cross-layer synthesis
    synthesis: Optional[Dict] = None


def run_layer_analysis(
    model,
    layer_idx: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    analysis_config: AnalysisConfig = ANALYSIS_CONFIG,
    weight_extractor: Optional[WeightExtractor] = None,
    sae_manager: Optional[SAEManager] = None,
    rope_analyzer: Optional[RoPEAnalyzer] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> LayerAnalysisResult:
    """
    Run complete analysis for a single layer.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing Layer {layer_idx}")
        print(f"{'='*60}")
    
    # Initialize managers if not provided
    if weight_extractor is None:
        weight_extractor = WeightExtractor(model, config)
    if sae_manager is None:
        sae_manager = SAEManager(config)
    if rope_analyzer is None:
        rope_analyzer = RoPEAnalyzer(config)
    
    # Extract weights
    if verbose:
        print("  Loading weights...")
    layer_weights = weight_extractor.get_layer(layer_idx, device="cpu", dtype=torch.float32)
    
    # Load SAE features
    # IMPORTANT: SAE layer may differ from attention layer!
    # Post-MLP residual of block L feeds attention in block L+1,
    # so we may need to use SAE from layer L-1 for attention block L.
    sae_layer = config.get_sae_layer_for_attn(layer_idx)
    if verbose:
        if sae_layer != layer_idx:
            print(f"  Loading SAE features from layer {sae_layer} (offset={config.sae_layer_offset_for_attn})...")
        else:
            print("  Loading SAE features...")
    sae_features = sae_manager.get_features(sae_layer, device=device, dtype=dtype)
    
    n_heads = config.num_attention_heads
    is_sliding = config.is_sliding_window_layer(layer_idx)
    
    routing_results = []
    writing_results = []
    program_results = []
    rope_stability = []
    
    for qh in range(n_heads):
        if verbose:
            print(f"  Analyzing head {qh}...")
        
        kv_group = config.query_to_kv_group(qh)
        W_Q = layer_weights.W_Q[qh].to(device=device, dtype=dtype)
        W_K = layer_weights.W_K[kv_group].to(device=device, dtype=dtype)
        
        # QK Routing
        routing = analyze_head_routing(
            sae_features, W_Q, W_K,
            layer_idx, qh, kv_group,
            config, analysis_config
        )
        routing_results.append(routing)
        
        # OV Writing
        writing = analyze_head_writing(
            sae_features, layer_weights, qh, config
        )
        writing_results.append(writing)
        
        # Feature Programs
        programs = analyze_head_programs(routing, writing)
        program_results.append(programs)
        
        # RoPE Stability
        Q_f, K_f = project_to_qk_space(sae_features.decoder_subset.to(device), W_Q, W_K)
        deltas = analysis_config.delta_positions
        if is_sliding:
            max_delta = config.sliding_window_size
            deltas = [d for d in deltas if d <= max_delta]
        
        stability = rope_analyzer.compute_stability_curve(Q_f, K_f, deltas)
        rope_stability.append({
            "query_head": qh,
            **stability
        })
    
    # Cross-head redundancy
    redundancy = compute_cross_head_redundancy(routing_results)
    
    # Head contribution ranking (by OV write norm)
    contributions = [(i, r.metrics.write_norm_mean) for i, r in enumerate(writing_results)]
    contributions.sort(key=lambda x: x[1], reverse=True)
    head_ranking = [c[0] for c in contributions]
    
    # Convert to serializable dicts
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            d = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, torch.Tensor):
                    continue  # Skip tensors
                elif hasattr(v, 'value'):  # Enum
                    d[k] = v.value
                elif isinstance(v, list):
                    d[k] = [to_dict(item) if hasattr(item, '__dict__') else item for item in v]
                elif isinstance(v, dict):
                    d[k] = {kk: to_dict(vv) if hasattr(vv, '__dict__') else vv for kk, vv in v.items()}
                else:
                    d[k] = v
            return d
        return obj
    
    return LayerAnalysisResult(
        layer_idx=layer_idx,
        is_sliding_window=is_sliding,
        sae_layer_used=sae_layer,
        routing_results=[to_dict(r) for r in routing_results],
        writing_results=[to_dict(r) for r in writing_results],
        program_results=[to_dict(r) for r in program_results],
        rope_stability=rope_stability,
        cross_head_redundancy={str(k): v for k, v in redundancy.items()},
        head_contribution_ranking=head_ranking,
    )


def run_full_analysis(
    model,
    config: Gemma2Config = GEMMA2_CONFIG,
    analysis_config: AnalysisConfig = ANALYSIS_CONFIG,
    device: str = "cuda",
    verbose: bool = True,
) -> AnalysisRun:
    """
    Run complete analysis across all configured layers.
    """
    if verbose:
        print_config_summary(config)
        print(f"\nLayers to analyze: {analysis_config.layers_to_analyze}")
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Shared managers
    weight_extractor = WeightExtractor(model, config)
    sae_manager = SAEManager(config)
    rope_analyzer = RoPEAnalyzer(config)
    
    layer_results = {}
    
    for layer_idx in analysis_config.layers_to_analyze:
        try:
            result = run_layer_analysis(
                model, layer_idx, config, analysis_config,
                weight_extractor, sae_manager, rope_analyzer,
                device=device, verbose=verbose
            )
            layer_results[layer_idx] = result
        except Exception as e:
            print(f"Error analyzing layer {layer_idx}: {e}")
            continue
    
    run = AnalysisRun(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else {},
        analysis_config=asdict(analysis_config) if hasattr(analysis_config, '__dataclass_fields__') else {},
        layers_analyzed=list(layer_results.keys()),
        layer_results=layer_results,
    )
    
    return run


def save_results(run: AnalysisRun, output_dir: str = "./analysis_outputs"):
    """Save analysis results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"analysis_{run.run_id}.json")
    
    # Custom serializer for dataclasses
    def serialize(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: serialize(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return obj
    
    with open(filepath, 'w') as f:
        json.dump(serialize(run), f, indent=2, default=str)
    
    print(f"Results saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    print("Analysis Pipeline module loaded.")
    print("Use run_full_analysis(model) to run complete analysis.")
