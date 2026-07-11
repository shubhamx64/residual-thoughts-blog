"""
Run Intervention Experiments for Gemma-2 Attention Head Ablation.

Orchestrates the full experiment:
1. Load model and tokenizer
2. Generate or load prompts
3. Run baseline and ablation conditions
4. Record accuracy and logprob margins
5. Save results to JSON

Usage:
    # Generate prompts and run baseline
    python run_intervention.py --generate_prompts --condition baseline
    
    # Run ablation experiment
    python run_intervention.py --condition ablate --ablate_layer 23 --ablate_head 1
    
    # Run all conditions
    python run_intervention.py --run_all
"""

import torch
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from intervention_prompts import (
    PromptConfig,
    GeneratedPrompt,
    generate_prompt_set,
    save_prompts,
    load_prompts,
)
from head_ablation import HeadAblator, ablate_head, verify_ablation, get_head_norm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InterventionConfig:
    """Configuration for intervention experiments."""
    
    # Model
    model_id: str = "google/gemma-2-2b"
    device: str = "cuda"
    dtype: str = "bfloat16"  # or "float16"
    
    # Target head (configurable)
    target_layer: int = 23
    target_head: int = 1
    
    # Control heads
    same_layer_control_heads: List[int] = field(default_factory=lambda: [0, 2])
    neighbor_layer_controls: List[Tuple[int, int]] = field(
        default_factory=lambda: [(21, 1), (25, 1)]
    )
    
    # Prompt configuration
    prompts_file: str = "intervention_prompts.json"
    
    # Generation settings
    temperature: float = 0.0  # Small value for near-deterministic sampling
    max_new_tokens: int = 5  # Just need the choice letter
    
    # Output
    results_dir: str = "./intervention_results"
    
    # Gemma-2 specific
    head_dim: int = 256


@dataclass
class PromptResult:
    """Result for a single prompt."""
    prompt_id: int
    filler_word_count: int
    
    # Ground truth
    correct_choice: str
    correct_value: int
    
    # Model output
    generated_text: str
    predicted_choice: str
    is_correct: bool
    
    # Logprobs (if available)
    logprobs: Dict[str, float] = field(default_factory=dict)
    margin: Optional[float] = None  # logp(correct) - logp(best_wrong)
    
    # Token distance (actual position delta, not word count)
    token_distance: Optional[int] = None  # tokens from last table token to Answer:
    
    # Symmetry control
    order: str = "original"  # 'original' or 'flipped'
    pair_id: Optional[int] = None  # ID of paired prompt


@dataclass
class ConditionResult:
    """Results for an experimental condition."""
    condition_name: str
    ablated_heads: List[Tuple[int, int]]  # [(layer, head), ...]
    
    # Summary stats
    overall_accuracy: float
    accuracy_by_filler: Dict[int, float]  # filler_words -> accuracy
    mean_margin: float
    margin_by_filler: Dict[int, float]
    
    # A/B prediction bias diagnostic
    a_fraction: float = 0.0  # Fraction of predictions that are 'A'
    b_fraction: float = 0.0  # Fraction of predictions that are 'B'
    a_fraction_by_filler: Dict[int, float] = field(default_factory=dict)
    
    # Per-prompt results
    prompt_results: List[PromptResult] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    n_prompts: int = 0


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(config: InterventionConfig):
    """Load Gemma-2 model and tokenizer."""
    
    print(f"Loading model: {config.model_id}")
    
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map=config.device,
    )
    model.eval()
    
    print(f"Model loaded on {config.device} with dtype {config.dtype}")
    
    return model, tokenizer


# =============================================================================
# Inference
# =============================================================================

# Track if we've validated tokenization (only need to do once)
_tokenization_validated = False

def validate_choice_tokenization(tokenizer, choices: List[str] = ['A', 'B', 'C', 'D']):
    """
    Validate that each choice letter is a single token.
    
    This is critical for logits-only scoring - if 'A' becomes multiple tokens,
    our logprob comparison is wrong.
    """
    global _tokenization_validated
    if _tokenization_validated:
        return
    
    print("\n" + "=" * 60)
    print("TOKENIZATION VALIDATION (run once at startup)")
    print("=" * 60)
    
    for choice in choices:
        # Try both with and without leading space
        tokens_no_space = tokenizer.encode(choice, add_special_tokens=False)
        tokens_with_space = tokenizer.encode(" " + choice, add_special_tokens=False)
        
        print(f"  '{choice}' -> tokens: {tokens_no_space} (len={len(tokens_no_space)})")
        print(f"  ' {choice}' -> tokens: {tokens_with_space} (len={len(tokens_with_space)})")
        
        if len(tokens_no_space) != 1:
            raise ValueError(f"Choice '{choice}' tokenizes to {len(tokens_no_space)} tokens, expected 1!")
    
    print("  ✓ All choices are single tokens")
    print("=" * 60 + "\n")
    _tokenization_validated = True


def get_choice_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    choices: List[str] = ['A', 'B', 'C', 'D'],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Get log probabilities for each choice token at the next-token position.
    
    IMPORTANT: The prompt should end with 'Answer: ' (WITH trailing space).
    This way we score the logprob for plain 'A' and 'B' tokens.
    
    Returns dict mapping choice letter to its logprob.
    """
    # Validate tokenization once
    validate_choice_tokenization(tokenizer, choices)
    
    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # logits shape: [1, seq_len, vocab_size]
        # We want the logits for predicting the NEXT token after the last input token
        logits = outputs.logits[0, -1, :]  # [vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get logprobs for each choice token
    choice_logprobs = {}
    for choice in choices:
        # Tokenize the choice letter (we validated it's 1 token above)
        choice_tokens = tokenizer.encode(choice, add_special_tokens=False)
        token_id = choice_tokens[0]
        choice_logprobs[choice] = log_probs[token_id].item()
    
    return choice_logprobs


def generate_choice(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 5,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """
    Generate model's response (greedy decoding).
    
    Returns the generated text (should be just the choice letter).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
    
    # Decode only the new tokens
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()


def extract_choice(generated_text: str, valid_choices: List[str] = ['A', 'B', 'C', 'D']) -> str:
    """
    Extract the choice letter from generated text.
    
    Handles formats like "A", "A)", "A.", etc.
    """
    text = generated_text.strip().upper()
    
    # Direct match
    if text in valid_choices:
        return text
    
    # First character
    if len(text) > 0 and text[0] in valid_choices:
        return text[0]
    
    # Search for any valid choice
    for choice in valid_choices:
        if choice in text:
            return choice
    
    return "INVALID"


def run_single_prompt(
    model,
    tokenizer,
    prompt: GeneratedPrompt,
    config: InterventionConfig,
) -> PromptResult:
    """
    Run a single prompt using LOGITS-ONLY evaluation.
    
    No generation or parsing - just:
    1. Get logits at the Answer: position
    2. Compare logp(A) vs logp(B)
    3. predicted_choice = argmax
    4. margin = logp(correct) - logp(incorrect)
    
    This is deterministic and avoids all generation/parsing artifacts.
    """
    
    valid_choices = ['A', 'B', 'C', 'D'][:prompt.n_choices]
    
    # Get logprobs directly (this is the ONLY inference we do)
    logprobs = get_choice_logprobs(
        model, tokenizer, prompt.prompt_text,
        choices=valid_choices,
        device=config.device,
    )
    
    # Predicted choice = argmax over logprobs
    predicted_choice = max(logprobs, key=logprobs.get) if logprobs else "INVALID"
    is_correct = (predicted_choice == prompt.correct_choice)
    
    # Compute margin: logp(correct) - logp(best_wrong)
    margin = None
    if prompt.correct_choice in logprobs:
        correct_logp = logprobs[prompt.correct_choice]
        wrong_logps = [lp for c, lp in logprobs.items() if c != prompt.correct_choice]
        if wrong_logps:
            best_wrong_logp = max(wrong_logps)
            margin = correct_logp - best_wrong_logp
    
    # Compute token distance: from end of prefix (table) to end of prompt (Answer:)
    # This is the actual distance the model needs to "remember" across
    token_distance = None
    try:
        # Use prefix from prompt object (if available from new generator)
        if hasattr(prompt, 'prefix') and prompt.prefix:
            prefix_tokens = tokenizer.encode(prompt.prefix, add_special_tokens=False)
            full_tokens = tokenizer.encode(prompt.prompt_text, add_special_tokens=False)
            token_distance = len(full_tokens) - len(prefix_tokens)
        else:
            # Fallback for old prompts: find DATA end
            prompt_str = prompt.prompt_text
            # Find end of table (after last VALUE line, before filler or query)
            data_marker = "DATA:\n"
            if data_marker in prompt_str:
                data_start = prompt_str.find(data_marker) + len(data_marker)
                # Find where filler or query starts
                filler_marker = "[FILLER START]"
                query_marker = "Query:"
                
                if filler_marker in prompt_str:
                    data_end = prompt_str.find(filler_marker)
                else:
                    data_end = prompt_str.find(query_marker)
                
                prefix = prompt_str[:data_end]
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                full_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
                token_distance = len(full_tokens) - len(prefix_tokens)
    except Exception:
        pass  # Don't fail if we can't compute distance
    
    # For logging/debugging, show what the model would have output
    # (but this is NOT used for accuracy - only logprobs matter)
    generated_text = f"[logits-only: argmax={predicted_choice}]"
    
    return PromptResult(
        prompt_id=prompt.prompt_id,
        filler_word_count=prompt.filler_word_count,
        correct_choice=prompt.correct_choice,
        correct_value=prompt.correct_value,
        generated_text=generated_text,
        predicted_choice=predicted_choice,
        is_correct=is_correct,
        logprobs=logprobs,
        margin=margin,
        token_distance=token_distance,
        order=getattr(prompt, 'order', 'original'),
        pair_id=getattr(prompt, 'pair_id', None),
    )


# =============================================================================
# Experiment Conditions
# =============================================================================

def run_condition(
    model,
    tokenizer,
    prompts: List[GeneratedPrompt],
    config: InterventionConfig,
    condition_name: str,
    ablate_targets: List[Tuple[int, int]] = None,
    scale: float = 0.0,  # Scale factor for ablation (0.0 = zero, 0.5 = half, 1.0 = full)
) -> ConditionResult:
    """
    Run all prompts under a specific condition.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompts: List of prompts to run
        config: Experiment configuration
        condition_name: Name for this condition (e.g., "baseline", "ablate_L23H1")
        ablate_targets: List of (layer, head) tuples to ablate, or None for baseline
        scale: Scale factor for ablation (0.0 = zero out, 1.0 = keep full)
    """
    
    ablate_targets = ablate_targets or []
    
    print(f"\n{'=' * 60}")
    print(f"Running condition: {condition_name}")
    if ablate_targets:
        scale_str = f" (scale={scale})" if scale != 0.0 else ""
        print(f"Ablating: {[f'L{l}H{h}' for l, h in ablate_targets]}{scale_str}")
    else:
        print("No ablation (baseline)")
    print(f"{'=' * 60}")
    
    # Set up ablators
    ablators = []
    for layer, head in ablate_targets:
        ablator = HeadAblator(model, layer, head, config.head_dim)
        ablator.ablate(scale=scale)
        ablators.append(ablator)
        
        # Verify ablation (only if fully zeroed)
        if scale == 0.0:
            if verify_ablation(model, layer, head, config.head_dim):
                print(f"  ✓ L{layer}H{head} ablation verified (norm = 0)")
            else:
                norm = get_head_norm(model, layer, head, config.head_dim)
                print(f"  ✗ L{layer}H{head} ablation FAILED (norm = {norm:.4f})")
    
    # Run prompts
    results = []
    for prompt in tqdm(prompts, desc=condition_name):
        result = run_single_prompt(model, tokenizer, prompt, config)
        results.append(result)
    
    # Restore ablated heads
    for ablator in ablators:
        ablator.restore()
    
    # Compute stats
    overall_accuracy = sum(r.is_correct for r in results) / len(results)
    
    # Group by filler length
    filler_groups: Dict[int, List[PromptResult]] = {}
    for r in results:
        if r.filler_word_count not in filler_groups:
            filler_groups[r.filler_word_count] = []
        filler_groups[r.filler_word_count].append(r)
    
    accuracy_by_filler = {
        filler: sum(r.is_correct for r in group) / len(group)
        for filler, group in filler_groups.items()
    }
    
    margins = [r.margin for r in results if r.margin is not None]
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    
    margin_by_filler = {}
    for filler, group in filler_groups.items():
        group_margins = [r.margin for r in group if r.margin is not None]
        if group_margins:
            margin_by_filler[filler] = sum(group_margins) / len(group_margins)
    
    # A/B prediction bias diagnostic
    a_count = sum(1 for r in results if r.predicted_choice == 'A')
    b_count = sum(1 for r in results if r.predicted_choice == 'B')
    a_fraction = a_count / len(results) if results else 0.0
    b_fraction = b_count / len(results) if results else 0.0
    
    a_fraction_by_filler = {}
    for filler, group in filler_groups.items():
        a_in_group = sum(1 for r in group if r.predicted_choice == 'A')
        a_fraction_by_filler[filler] = a_in_group / len(group) if group else 0.0
    
    return ConditionResult(
        condition_name=condition_name,
        ablated_heads=ablate_targets,
        overall_accuracy=overall_accuracy,
        accuracy_by_filler=accuracy_by_filler,
        mean_margin=mean_margin,
        margin_by_filler=margin_by_filler,
        a_fraction=a_fraction,
        b_fraction=b_fraction,
        a_fraction_by_filler=a_fraction_by_filler,
        prompt_results=results,
        timestamp=datetime.now().isoformat(),
        n_prompts=len(prompts),
    )


# =============================================================================
# Main Experiment Runner
# =============================================================================

def save_results(results: List[ConditionResult], output_dir: str, filename: str):
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        'n_conditions': len(results),
        'conditions': [],
    }
    
    for r in results:
        condition_data = {
            'condition_name': r.condition_name,
            'ablated_heads': r.ablated_heads,
            'overall_accuracy': r.overall_accuracy,
            'accuracy_by_filler': {str(k): v for k, v in r.accuracy_by_filler.items()},
            'mean_margin': r.mean_margin,
            'margin_by_filler': {str(k): v for k, v in r.margin_by_filler.items()},
            'timestamp': r.timestamp,
            'n_prompts': r.n_prompts,
            'prompt_results': [asdict(pr) for pr in r.prompt_results],
        }
        data['conditions'].append(condition_data)
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


def print_summary(results: List[ConditionResult]):
    """Print summary of experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Header
    filler_lengths = sorted(results[0].accuracy_by_filler.keys())
    header = f"{'Condition':<25} {'Overall':>8}"
    for filler in filler_lengths:
        header += f" {filler:>6}w"
    header += f" {'Margin':>8}"
    print(header)
    print("-" * 80)
    
    # Results
    for r in results:
        row = f"{r.condition_name:<25} {r.overall_accuracy:>7.1%}"
        for filler in filler_lengths:
            acc = r.accuracy_by_filler.get(filler, 0)
            row += f" {acc:>6.1%}"
        row += f" {r.mean_margin:>8.2f}"
        print(row)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run intervention experiments")
    
    # Mode
    parser.add_argument('--generate_prompts', action='store_true',
                        help='Generate new prompts before running')
    parser.add_argument('--prompts_file', type=str, default='intervention_prompts.json',
                        help='Path to prompts file')
    
    # Condition selection
    parser.add_argument('--condition', type=str, default='baseline',
                        choices=['baseline', 'ablate', 'control_same_layer', 'control_neighbor'],
                        help='Which condition to run')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all conditions')
    parser.add_argument('--head_sweep', action='store_true',
                        help='Run ablation sweep across specified heads')
    parser.add_argument('--sweep_heads', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated heads to sweep (default: 0,1,2,3,4,5,6,7)')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip baseline run in head_sweep mode')
    parser.add_argument('--dose_response', action='store_true',
                        help='Run dose-response for --dose_head with scales 1.0, 0.5, 0.0')
    parser.add_argument('--dose_head', type=int, default=0,
                        help='Head to run dose-response on (default: 0)')
    parser.add_argument('--dose_scales', type=str, default='1.0,0.5,0.0',
                        help='Comma-separated scale factors for dose-response')
    
    # Ablation targets
    parser.add_argument('--ablate_layer', type=int, default=23,
                        help='Layer to ablate (for ablate condition)')
    parser.add_argument('--ablate_head', type=int, default=1,
                        help='Single head to ablate (for ablate condition)')
    parser.add_argument('--heads', type=str, default=None,
                        help='Comma-separated heads to ablate together, e.g., --layer 23 --heads 0,1,2')
    
    # Control targets
    parser.add_argument('--control_head', type=int, default=0,
                        help='Control head in same layer')
    parser.add_argument('--neighbor_layer', type=int, default=21,
                        help='Neighbor layer for control')
    
    # Prompt generation settings
    parser.add_argument('--n_prompts', type=int, default=50,
                        help='Prompts per filler length')
    parser.add_argument('--filler_lengths', type=str, default='0,100,200,400,600,800',
                        help='Comma-separated filler word counts')
    
    # Output
    parser.add_argument('--results_dir', type=str, default='./intervention_results',
                        help='Directory for results')
    
    # Generation settings
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='Sampling temperature (0.01 = near-deterministic)')
    parser.add_argument('--max_new_tokens', type=int, default=3,
                        help='Maximum new tokens to generate')
    
    # Symmetry control: filter by prompt order
    parser.add_argument('--order', type=str, default='all', choices=['original', 'flipped', 'all'],
                        help='Filter prompts by order type: original, flipped, or all')
    
    args = parser.parse_args()
    
    config = InterventionConfig(
        target_layer=args.ablate_layer,
        target_head=args.ablate_head,
        prompts_file=args.prompts_file,
        results_dir=args.results_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Generate prompts if requested
    if args.generate_prompts:
        filler_lengths = [int(x) for x in args.filler_lengths.split(',')]
        prompt_config = PromptConfig(
            n_prompts_per_filler_length=args.n_prompts,
            filler_word_counts=filler_lengths,
            output_file=args.prompts_file,
        )
        prompts = generate_prompt_set(prompt_config)
        save_prompts(prompts, args.prompts_file)
    
    # Load prompts
    print(f"\nLoading prompts from: {args.prompts_file}")
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Filter by order if requested (symmetry control)
    if args.order != 'all':
        original_count = len(prompts)
        prompts = [p for p in prompts if getattr(p, 'order', 'original') == args.order]
        print(f"Filtered to {len(prompts)} {args.order} prompts (from {original_count})")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    # HEAD SWEEP MODE: run ablation sweep across specified heads
    if args.head_sweep:
        layer = args.ablate_layer
        heads_to_sweep = [int(h) for h in args.sweep_heads.split(',')]
        
        print(f"\n{'='*60}")
        print(f"HEAD SWEEP: Layer {layer}, heads {heads_to_sweep}")
        if args.no_baseline:
            print("(Skipping baseline)")
        print(f"{'='*60}")
        
        # Build conditions list
        conditions_to_run = []
        if not args.no_baseline:
            conditions_to_run.append(('baseline', []))
        for h in heads_to_sweep:
            conditions_to_run.append((f'L{layer}H{h}', [(layer, h)]))
        
        results = []
        for cond_name, ablate_targets in conditions_to_run:
            result = run_condition(
                model, tokenizer, prompts, config,
                condition_name=cond_name,
                ablate_targets=ablate_targets,
            )
            results.append(result)
        
        # Print sweep summary with token distance stats
        print("\n" + "=" * 80)
        print("HEAD SWEEP SUMMARY")
        print("=" * 80)
        
        # Get token distances from first result
        sample_result = results[0]
        token_dists = [r.token_distance for r in sample_result.prompt_results if r.token_distance]
        if token_dists:
            print(f"\nToken distances: min={min(token_dists)}, median={sorted(token_dists)[len(token_dists)//2]}, max={max(token_dists)}")
            
            # Group by filler bin
            from collections import defaultdict
            dist_by_filler = defaultdict(list)
            for r in sample_result.prompt_results:
                if r.token_distance:
                    dist_by_filler[r.filler_word_count].append(r.token_distance)
            
            print("\nToken distance by filler bin:")
            for filler in sorted(dist_by_filler.keys()):
                dists = dist_by_filler[filler]
                print(f"  {filler}w: min={min(dists)}, median={sorted(dists)[len(dists)//2]}, max={max(dists)}")
        
        # Print comparison table
        baseline = results[0]
        filler_lengths = sorted(baseline.accuracy_by_filler.keys())
        
        print("\n" + "-" * 80)
        print(f"{'Condition':<12} {'Overall':>8} {'Δ':>6}", end="")
        for filler in filler_lengths:
            print(f" {filler:>6}w", end="")
        print(f" {'Margin':>8} {'Δmargin':>8}")
        print("-" * 80)
        
        for r in results:
            delta_acc = r.overall_accuracy - baseline.overall_accuracy
            delta_margin = r.mean_margin - baseline.mean_margin
            
            print(f"{r.condition_name:<12} {r.overall_accuracy:>7.1%} {delta_acc:>+5.1%}", end="")
            for filler in filler_lengths:
                acc = r.accuracy_by_filler.get(filler, 0)
                print(f" {acc:>6.1%}", end="")
            print(f" {r.mean_margin:>8.2f} {delta_margin:>+7.2f}")
        
        print("=" * 80)
        
        # Print which head hurts most at longest distance
        longest_filler = max(filler_lengths)
        print(f"\nΔaccuracy at longest distance ({longest_filler}w):")
        baseline_long_acc = baseline.accuracy_by_filler.get(longest_filler, 0)
        for r in results[1:]:  # Skip baseline
            r_long_acc = r.accuracy_by_filler.get(longest_filler, 0)
            delta = r_long_acc - baseline_long_acc
            print(f"  {r.condition_name}: {delta:+.1%}")
        
        # A/B BIAS DIAGNOSTIC
        print("\n" + "-" * 80)
        print("A/B PREDICTION BIAS (fraction predicting 'A')")
        print("-" * 80)
        print(f"{'Condition':<12} {'Overall':>8}", end="")
        for filler in filler_lengths:
            print(f" {filler:>6}w", end="")
        print()
        
        for r in results:
            print(f"{r.condition_name:<12} {r.a_fraction:>7.1%}", end="")
            for filler in filler_lengths:
                a_frac = r.a_fraction_by_filler.get(filler, 0)
                print(f" {a_frac:>6.1%}", end="")
            print()
        
        print("-" * 80)
        print("(If ablation makes A/B ~0% or ~100%, the effect is just bias, not retrieval)")
        
        # ORIGINAL VS FLIPPED COMPARISON
        has_flipped = any(getattr(p, 'order', 'original') == 'flipped' for p in prompts)
        if has_flipped:
            print("\n" + "=" * 90)
            print("ORIGINAL VS FLIPPED COMPARISON")
            print("=" * 90)
            print(f"{'Condition':<12} {'Orig Acc':>9} {'Flip Acc':>9} {'Δacc':>7} {'Orig A%':>8} {'Flip A%':>8} {'ΔA%':>7}")
            print("-" * 90)
            
            for r in results:
                orig_res = [pr for pr in r.prompt_results if getattr(pr, 'order', 'original') == 'original']
                flip_res = [pr for pr in r.prompt_results if getattr(pr, 'order', 'original') == 'flipped']
                
                if orig_res and flip_res:
                    orig_acc = sum(1 for pr in orig_res if pr.is_correct) / len(orig_res)
                    flip_acc = sum(1 for pr in flip_res if pr.is_correct) / len(flip_res)
                    orig_a = sum(1 for pr in orig_res if pr.predicted_choice == 'A') / len(orig_res)
                    flip_a = sum(1 for pr in flip_res if pr.predicted_choice == 'A') / len(flip_res)
                    print(f"{r.condition_name:<12} {orig_acc:>8.1%} {flip_acc:>8.1%} {flip_acc-orig_acc:>+6.1%} "
                          f"{orig_a:>7.1%} {flip_a:>7.1%} {flip_a-orig_a:>+6.1%}")
            
            # PAIRED CONSISTENCY
            print("\n" + "-" * 90)
            print("PAIRED CONSISTENCY (original↔flipped pairs)")
            print("-" * 90)
            print(f"{'Condition':<12} {'Both✓':>8} {'Neither':>8} {'Orig✓':>8} {'Flip✓':>8} {'Consist':>8}")
            print("-" * 90)
            
            for r in results:
                orig_by_id = {}
                flip_by_pair = {}
                for pr in r.prompt_results:
                    order = getattr(pr, 'order', 'original')
                    if order == 'flipped':
                        pair_id = getattr(pr, 'pair_id', None)
                        if pair_id is not None:
                            flip_by_pair[pair_id] = pr
                    else:
                        orig_by_id[pr.prompt_id] = pr
                
                if not flip_by_pair:
                    continue
                
                both = neither = orig_only = flip_only = 0
                for oid, opr in orig_by_id.items():
                    if oid not in flip_by_pair:
                        continue
                    fpr = flip_by_pair[oid]
                    if opr.is_correct and fpr.is_correct:
                        both += 1
                    elif not opr.is_correct and not fpr.is_correct:
                        neither += 1
                    elif opr.is_correct:
                        orig_only += 1
                    else:
                        flip_only += 1
                
                total = both + neither + orig_only + flip_only
                if total > 0:
                    print(f"{r.condition_name:<12} {both/total:>7.1%} {neither/total:>7.1%} "
                          f"{orig_only/total:>7.1%} {flip_only/total:>7.1%} {(both+neither)/total:>7.1%}")
            
            print("-" * 90)
            print("Both✓=both correct, Consist=model invariant to label order")
            print("=" * 90)
    
    elif args.dose_response:
        # DOSE RESPONSE MODE: run same head with different scale factors
        layer = args.ablate_layer
        head = args.dose_head
        scales = [float(s) for s in args.dose_scales.split(',')]
        
        print(f"\n{'='*60}")
        print(f"DOSE RESPONSE: Layer {layer} Head {head}")
        print(f"Scales: {scales}")
        print(f"{'='*60}")
        
        results = []
        for scale in scales:
            if scale == 1.0:
                # scale=1.0 is baseline (no change)
                cond_name = 'baseline'
                ablate_targets = []
            else:
                cond_name = f'L{layer}H{head}_s{scale}'
                ablate_targets = [(layer, head)]
            
            result = run_condition(
                model, tokenizer, prompts, config,
                condition_name=cond_name,
                ablate_targets=ablate_targets,
                scale=scale,
            )
            results.append(result)
        
        # Print dose response summary
        print("\n" + "=" * 80)
        print("DOSE RESPONSE SUMMARY")
        print("=" * 80)
        
        baseline = results[0] if scales[0] == 1.0 else None
        filler_lengths = sorted(results[0].accuracy_by_filler.keys())
        
        print(f"\n{'Scale':<12} {'Overall':>8} {'Δ':>6}", end="")
        for filler in filler_lengths:
            print(f" {filler:>6}w", end="")
        print(f" {'Margin':>8}")
        print("-" * 80)
        
        for i, (scale, r) in enumerate(zip(scales, results)):
            if baseline:
                delta_acc = r.overall_accuracy - baseline.overall_accuracy
            else:
                delta_acc = 0.0
            
            print(f"{scale:<12} {r.overall_accuracy:>7.1%} {delta_acc:>+5.1%}", end="")
            for filler in filler_lengths:
                acc = r.accuracy_by_filler.get(filler, 0)
                print(f" {acc:>6.1%}", end="")
            print(f" {r.mean_margin:>8.2f}")
        
        print("=" * 80)
    
    elif args.heads:
        # MULTI-HEAD ABLATION MODE: ablate multiple heads together (no baseline)
        layer = args.ablate_layer
        heads = [int(h) for h in args.heads.split(',')]
        
        print(f"\n{'='*60}")
        print(f"MULTI-HEAD ABLATION: Layer {layer}, heads {heads}")
        print(f"{'='*60}")
        
        # Run multi-head ablation only
        ablate_targets = [(layer, h) for h in heads]
        heads_str = '+'.join(f'H{h}' for h in heads)
        result = run_condition(
            model, tokenizer, prompts, config,
            condition_name=f'L{layer}_{heads_str}',
            ablate_targets=ablate_targets,
        )
        
        results = [result]
        
        # Print summary
        print("\n" + "=" * 80)
        print("MULTI-HEAD ABLATION RESULT")
        print("=" * 80)
        
        filler_lengths = sorted(result.accuracy_by_filler.keys())
        
        print(f"\n{'Condition':<20} {'Acc':>8} {'Margin':>8} {'A%':>6}", end="")
        for filler in filler_lengths:
            print(f" {filler:>5}w", end="")
        print()
        print("-" * 80)
        
        print(f"{result.condition_name:<20} {result.overall_accuracy:>7.1%} {result.mean_margin:>8.2f} {result.a_fraction:>5.1%}", end="")
        for filler in filler_lengths:
            acc = result.accuracy_by_filler.get(filler, 0)
            print(f" {acc:>5.1%}", end="")
        print()
        
        print("=" * 80)
    
    else:
        # Normal mode: specific conditions
        all_conditions = {
            'baseline': [],
            'ablate': [(args.ablate_layer, args.ablate_head)],
            'control_same_layer': [(args.ablate_layer, args.control_head)],
            'control_neighbor': [(args.neighbor_layer, args.ablate_head)],
        }
        
        # Select conditions to run
        if args.run_all:
            conditions_to_run = list(all_conditions.keys())
        else:
            conditions_to_run = [args.condition]
        
        # Run experiments
        results = []
        for condition_name in conditions_to_run:
            ablate_targets = all_conditions[condition_name]
            
            # Create descriptive name
            if ablate_targets:
                desc = f"{condition_name}_L{ablate_targets[0][0]}H{ablate_targets[0][1]}"
            else:
                desc = condition_name
            
            result = run_condition(
                model, tokenizer, prompts, config,
                condition_name=desc,
                ablate_targets=ablate_targets,
            )
            results.append(result)
        
        # Print and save summary
        print_summary(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, args.results_dir, f"results_{timestamp}.json")


if __name__ == "__main__":
    main()
