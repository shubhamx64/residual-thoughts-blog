"""
Gemma-2 Weight-Space SAE Head Semantics Analysis

Main entry point for running the analysis pipeline.

Usage:
    python main.py [--layers 0,5,10,15,20,25] [--quick] [--no-hf-login]

The analysis characterizes what each attention head "wants to do" in feature
space using only weight matrices and SAE decoder directions.

Phases:
1. QK Routing: Who attends to whom?
2. OV Writing: What gets written when attending?
3. Feature Programs: Composed triplets (query→key→write)
4. RoPE Modulation: How does position affect routing?
5. Synthesis: Cross-layer patterns and steering recommendations
"""

import argparse
import os
import sys
import torch
from getpass import getpass

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Gemma2Config, AnalysisConfig, GEMMA2_CONFIG, ANALYSIS_CONFIG, print_config_summary
from analysis_pipeline import run_full_analysis, save_results
from synthesis import generate_synthesis_report


def setup_hf_login():
    """Handle HuggingFace login for gated models."""
    from huggingface_hub import login as hf_login
    try:
        tok = getpass("HF token (press Enter to skip): ")
        if tok.strip():
            hf_login(token=tok.strip())
            print("HF login: ok")
        else:
            print("HF login: skipped")
    except Exception as e:
        print(f"HF login: failed or skipped: {e}")


def load_model(config: Gemma2Config):
    """Load Gemma-2 model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {config.model_id}")
    print(f"Device: {config.device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.dtype,
        device_map="auto",
    )
    
    print(f"Model loaded successfully")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Gemma-2 Weight-Space SAE Head Semantics Analysis"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,5,10,15,20,25",
        help="Comma-separated list of layers to analyze (default: 0,5,10,15,20,25)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: analyze fewer layers (0,10,20)"
    )
    parser.add_argument(
        "--no-hf-login",
        action="store_true",
        help="Skip HuggingFace login prompt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=16384,
        help="Number of SAE features to analyze (default: 16384)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Parse layers
    if args.quick:
        layers = [2,3,12,13,24,25]
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]
    
    # Setup config
    config = Gemma2Config()
    if args.device:
        config.device = args.device
    config.feature_subset_size = args.subset_size
    
    analysis_config = AnalysisConfig(
        layers_to_analyze=layers,
        output_dir=args.output_dir,
    )
    
    # Print banner
    print("=" * 70)
    print("GEMMA-2 WEIGHT-SPACE SAE HEAD SEMANTICS ANALYSIS")
    print("=" * 70)
    print()
    
    # HF Login
    if not args.no_hf_login:
        setup_hf_login()
    
    # Load model
    print()
    model, tokenizer = load_model(config)
    print()
    
    # Run analysis
    print_config_summary(config)
    print(f"\nLayers to analyze: {layers}")
    print(f"Feature subset size: {config.feature_subset_size}")
    print()
    
    run = run_full_analysis(
        model,
        config=config,
        analysis_config=analysis_config,
        device=config.device,
        verbose=True,
    )
    
    # Generate synthesis
    print("\nGenerating synthesis report...")
    report = generate_synthesis_report(run)
    print()
    print(report)
    
    # Save results
    filepath = save_results(run, args.output_dir)
    
    # Save synthesis report
    report_path = filepath.replace(".json", "_synthesis.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Synthesis report saved to: {report_path}")
    
    print("\nAnalysis complete!")
    return run


if __name__ == "__main__":
    main()
