"""
Run Activation Validation for Phase 6.

Loads model, tokenizer, and SAE, then runs validation on target heads.
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Local imports
from config import AnalysisConfig
from sae_utils import SAEManager
from activation_validation import (
    ActivationValidator,
    ValidationConfig,
    load_validation_prompts,
)


def load_model_and_tokenizer(device: str = "cuda"):
    """Load Gemma-2 model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use Gemma2Config for model ID
    from config import Gemma2Config
    gemma_config = Gemma2Config()
    model_id = gemma_config.model_id
    
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        # Use eager attention to enable output_attentions=True
        attn_implementation="eager",
    )
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer

# =============================================================================
# TODO: Potential Optimizations for Future Work
# =============================================================================
# DONE: 1. RoPE correction added via --rope flag (per-distance B matrices)
# 2. Hook raw attention logits (before softmax) instead of attention weights
#    (current approach uses log(attention_weights) which is close enough)
# 3. Compare against a random baseline to ensure correlations are meaningful
# 4. Use more prompts and longer sequences for better distance bin coverage
# 5. Vectorize per-pair computation in validate_routing_rope_aware for speed
# 6. Use narrower distance bins or exact-distance RoPE instead of bin-center
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run activation validation")
    parser.add_argument(
        "analysis_json",
        nargs="?",
        help="Path to analysis JSON (optional, uses latest if not provided)"
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="6:3,15:0",
        help="Heads to validate in format layer:head,layer:head (default: 6:3,15:0)"
    )
    parser.add_argument(
        "--all-heads",
        action="store_true",
        help="Validate all heads for specified layers (use --layers to specify layers)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,5,6,10,15,20,25",
        help="Layers to use with --all-heads (default: 0,5,6,10,15,20,25)"
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        help="Number of prompts to use (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for validation report"
    )
    parser.add_argument(
        "--rope",
        action="store_true",
        help="Use RoPE-aware B matrices per distance bin (recommended)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="validation_prompts.csv",
        help="CSV file with prompts (default: validation_prompts.csv, use validation_prompts_short.csv for faster iteration)"
    )
    parser.add_argument(
        "--validate-writing",
        action="store_true",
        help="Also validate OV writing (W2F predicts feature deltas)"
    )
    parser.add_argument(
        "--writing-ablation",
        action="store_true",
        help="Use ablation-based ground truth for W2F validation (slower but more accurate)"
    )
    parser.add_argument(
        "--validate-ov",
        action="store_true",
        help="Validate OV_f matrix (attention output vectors in residual space, no SAE nonlinearity)"
    )
    parser.add_argument(
        "--sae-features",
        type=int,
        default=4096,
        help="Number of SAE features for QK routing validation (default: 4096, OV_f always uses full 16K)"
    )
    
    args = parser.parse_args()
    
    # Find analysis JSON
    if args.analysis_json:
        analysis_path = Path(args.analysis_json)
    else:
        # Find most recent analysis
        output_dir = Path("analysis_outputs")
        json_files = list(output_dir.glob("analysis_*.json"))
        if not json_files:
            print("No analysis JSON found. Run main.py first.")
            return
        analysis_path = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Using analysis: {analysis_path}")
    
    # Parse heads
    from config import Gemma2Config
    gemma_config = Gemma2Config()
    
    if args.all_heads:
        # Generate all heads for specified layers
        layers = [int(l.strip()) for l in args.layers.split(",")]
        heads = []
        for layer_idx in layers:
            for head_idx in range(gemma_config.num_attention_heads):
                heads.append((layer_idx, head_idx))
        print(f"Validating ALL {len(heads)} heads across layers {layers}")
    else:
        heads = []
        for head_str in args.heads.split(","):
            layer, head = map(int, head_str.strip().split(":"))
            heads.append((layer, head))
    
    print(f"Target heads: {heads}")
    print(f"Using {args.prompts} prompts")
    
    # Load analysis
    with open(analysis_path) as f:
        analysis = json.load(f)
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL AND SAE")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(args.device)
    
    # Load SAE - use Gemma2Config (SAEManager expects it)
    from config import Gemma2Config
    gemma_config = Gemma2Config()
    # Override feature subset size with CLI argument
    gemma_config.feature_subset_size = args.sae_features
    sae_manager = SAEManager(gemma_config, subset_size=args.sae_features)
    
    print(f"SAE features for QK routing: {args.sae_features}")
    print("SAE features for OV_f: 16384 (full)")
    
    # AnalysisConfig for other settings
    analysis_config = AnalysisConfig()
    
    # SAE layers will be auto-loaded on demand by sae_manager.encode()
    
    # Create validator
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION")
    print("=" * 60)
    
    val_config = ValidationConfig()
    validator = ActivationValidator(
        model=model,
        tokenizer=tokenizer,
        sae_manager=sae_manager,
        analysis_results=analysis,
        config=val_config,
        analysis_config=analysis_config,
        qk_features=args.sae_features,  # Pass feature count for report
    )
    
    # Run validation - load prompts from CSV
    all_prompts = load_validation_prompts(args.prompt_file)
    if not all_prompts:
        print(f"ERROR: No prompts loaded. Ensure {args.prompt_file} exists.")
        return
    prompts = all_prompts[:args.prompts]
    print(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}, using first {len(prompts)}")
    results = {"routing": {}}
    
    current_layer = None
    for layer_idx, head_idx in heads:
        # Clear CUDA memory when switching to a new layer
        if current_layer is not None and layer_idx != current_layer:
            print(f"\n  [Clearing CUDA memory after layer {current_layer}...]")
            validator.B_matrices.clear()  # Clear cached B matrices
            torch.cuda.empty_cache()
        current_layer = layer_idx
        
        print(f"\nValidating L{layer_idx}H{head_idx}...")
        if args.rope:
            result = validator.validate_routing_for_head_rope_aware(prompts, layer_idx, head_idx)
        else:
            result = validator.validate_routing_for_head(prompts, layer_idx, head_idx)
        results["routing"][f"L{layer_idx}H{head_idx}"] = result
        
        # Print comprehensive diagnostics
        print(f"  [SUMMARY]")
        print(f"    Pearson r:  {result.pearson_r:.4f}")
        print(f"    Spearman r: {result.spearman_r:.4f}")
        print(f"    Total pairs: {result.n_total_pairs:,}")
        
        print(f"  [DISTANCE-SPECIFIC SPEARMAN]")
        print(f"    Local  (0-4):     {result.spearman_local:.4f}")
        print(f"    Mid    (16-32):   {result.spearman_mid:.4f}")
        print(f"    Long   (128-256): {result.spearman_long:.4f}")
        print(f"    256+   (256+):    {result.spearman_256plus:.4f}")
        
        print(f"  [SIGN STABILITY]")
        print(f"    Sign stability: {result.sign_stability:.2%}")
        
        print(f"  [SKIPPED]")
        print(f"    Rows: {result.n_rows_skipped}/{result.n_rows_total} ({result.skipped_fraction:.2%})")
        if result.bins_with_no_data:
            print(f"    Empty bins: {', '.join(result.bins_with_no_data)}")
        
        # Print condensed per-bin table
        if result.pearson_by_distance:
            print(f"  [BY DISTANCE]")
            for bin_name in sorted(result.pearson_by_distance.keys(), key=lambda x: int(x.split('-')[0])):
                r_p = result.pearson_by_distance.get(bin_name, 0.0)
                r_s = result.spearman_by_distance.get(bin_name, 0.0)
                n = result.n_pairs_by_distance.get(bin_name, 0)
                pred_std = result.predicted_std_by_bin.get(bin_name, 0.0)
                act_std = result.actual_std_by_bin.get(bin_name, 0.0)
                print(f"    {bin_name:>10}: r_s={r_s:+.3f} r_p={r_p:+.3f} (n={n:,}, σ_pred={pred_std:.2f}, σ_act={act_std:.2f})")
    
    # === W2F Writing Validation (if requested) ===
    if args.validate_writing:
        print("\n" + "=" * 60)
        print("OV WRITING VALIDATION (Does W2F predict feature deltas?)")
        print("=" * 60)
        
        results["writing"] = {}
        
        for layer_idx, head_idx in heads:
            print(f"\nValidating W2F for L{layer_idx}H{head_idx}...")
            if args.writing_ablation:
                print("  (using ablation-based ground truth)")
                w_result = validator.validate_writing_for_head_ablation(prompts, layer_idx, head_idx)
            else:
                w_result = validator.validate_writing_for_head(prompts, layer_idx, head_idx)
            results["writing"][f"L{layer_idx}H{head_idx}"] = w_result
            
            print(f"  [W2F SUMMARY]")
            print(f"    Jaccard@{w_result.top_k}:     {w_result.jaccard_at_k:.4f}")
            print(f"    Top-K overlap:   {w_result.ndcg_at_k:.4f}")
            print(f"    Rank Spearman ρ: {w_result.delta_spearman_r:.4f}")
            print(f"    Positions:       {w_result.n_positions:,}")
        
        # Print aggregate W2F summary
        all_jaccard = [r.jaccard_at_k for r in results["writing"].values()]
        all_overlap = [r.ndcg_at_k for r in results["writing"].values()]
        all_spearman = [r.delta_spearman_r for r in results["writing"].values()]
        
        print(f"\n  [W2F AGGREGATE]")
        print(f"    Mean Jaccard@K:     {np.mean(all_jaccard):.4f} (std={np.std(all_jaccard):.4f})")
        print(f"    Mean Top-K overlap: {np.mean(all_overlap):.4f} (std={np.std(all_overlap):.4f})")
        print(f"    Mean Rank ρ:        {np.mean(all_spearman):.4f} (std={np.std(all_spearman):.4f})")
    
    # === OV_f Validation (if requested) ===
    if args.validate_ov:
        print("\n" + "=" * 60)
        print("OV_f VALIDATION (Does OV_f predict attention output vectors?)")
        print("Comparing in RESIDUAL SPACE (no SAE nonlinearity)")
        print("=" * 60)
        
        results["ov_f"] = {}
        
        # Group heads by layer for optimized batch processing
        from collections import defaultdict
        heads_by_layer = defaultdict(list)
        for layer_idx, head_idx in heads:
            heads_by_layer[layer_idx].append(head_idx)
        
        # Process layer by layer with cached baseline
        for layer_idx in sorted(heads_by_layer.keys()):
            layer_heads = heads_by_layer[layer_idx]
            
            if len(layer_heads) > 1:
                # Use optimized layer-batched validation (caches baseline)
                print(f"\nValidating OV_f for L{layer_idx} ({len(layer_heads)} heads, cached baseline)...")
                layer_results = validator.validate_OV_f_for_layer(prompts, layer_idx, layer_heads)
                
                for head_idx, ov_result in layer_results.items():
                    results["ov_f"][f"L{layer_idx}H{head_idx}"] = ov_result
                    print(f"  L{layer_idx}H{head_idx}: Cosine sim = {ov_result.jaccard_at_k:.4f}")
            else:
                # Single head - use standard method
                head_idx = layer_heads[0]
                print(f"\nValidating OV_f for L{layer_idx}H{head_idx}...")
                ov_result = validator.validate_OV_f_for_head(prompts, layer_idx, head_idx)
                results["ov_f"][f"L{layer_idx}H{head_idx}"] = ov_result
                print(f"  [OV_f SUMMARY]")
                print(f"    Cosine similarity: {ov_result.jaccard_at_k:.4f}")
                print(f"    Positions:         {ov_result.n_positions:,}")
        
        # Print aggregate OV_f summary
        all_cosine = [r.jaccard_at_k for r in results["ov_f"].values()]
        
        print(f"\n  [OV_f AGGREGATE]")
        print(f"    Mean Cosine Sim: {np.mean(all_cosine):.4f} (std={np.std(all_cosine):.4f})")
        print(f"    Min:  {np.min(all_cosine):.4f}")
        print(f"    Max:  {np.max(all_cosine):.4f}")
    
    # Generate report
    report = validator.generate_report(results)
    print("\n" + report)
    
    # Save report
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"validation_report_{timestamp}.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
