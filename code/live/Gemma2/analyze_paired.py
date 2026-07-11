"""
Paired Analysis for Intervention Experiments.

Computes paired statistics (Δmargin, Δaccuracy) and generates plots.

Usage:
    python analyze_paired.py --results_dir intervention_results --output_dir intervention_outputs
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PairedStats:
    """Paired statistics for comparing two conditions."""
    condition_name: str
    baseline_name: str
    
    # Overall
    delta_accuracy: float
    delta_margin: float
    
    # By filler bin
    delta_accuracy_by_filler: Dict[int, float]
    delta_margin_by_filler: Dict[int, float]
    
    # Bootstrap CIs for delta_margin
    delta_margin_ci_by_filler: Dict[int, Tuple[float, float]]
    
    # A/B fractions
    baseline_a_fraction: float
    condition_a_fraction: float
    
    # Sample size
    n_prompts: int


def load_results(results_file: str) -> Dict:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return (lower, upper)


def compute_paired_stats(baseline_results: Dict, condition_results: Dict) -> PairedStats:
    """
    Compute paired statistics between baseline and ablation condition.
    
    Uses prompt_id matching for paired comparisons.
    """
    # Build lookup by prompt_id
    baseline_by_id = {r['prompt_id']: r for r in baseline_results['prompt_results']}
    condition_by_id = {r['prompt_id']: r for r in condition_results['prompt_results']}
    
    # Compute paired deltas
    delta_margins = []
    delta_correct = []
    
    delta_by_filler = defaultdict(list)
    correct_by_filler_baseline = defaultdict(list)
    correct_by_filler_condition = defaultdict(list)
    
    for prompt_id, baseline_r in baseline_by_id.items():
        if prompt_id not in condition_by_id:
            continue
        
        condition_r = condition_by_id[prompt_id]
        filler = baseline_r['filler_word_count']
        
        # Paired margin delta
        b_margin = baseline_r.get('margin') or 0
        c_margin = condition_r.get('margin') or 0
        delta = c_margin - b_margin
        delta_margins.append(delta)
        delta_by_filler[filler].append(delta)
        
        # Paired accuracy
        b_correct = 1 if baseline_r['is_correct'] else 0
        c_correct = 1 if condition_r['is_correct'] else 0
        delta_correct.append(c_correct - b_correct)
        correct_by_filler_baseline[filler].append(b_correct)
        correct_by_filler_condition[filler].append(c_correct)
    
    # Aggregate stats
    fillers = sorted(delta_by_filler.keys())
    
    delta_accuracy_by_filler = {}
    delta_margin_by_filler = {}
    delta_margin_ci_by_filler = {}
    
    for filler in fillers:
        deltas = np.array(delta_by_filler[filler])
        delta_margin_by_filler[filler] = np.mean(deltas)
        delta_margin_ci_by_filler[filler] = bootstrap_ci(deltas)
        
        b_acc = np.mean(correct_by_filler_baseline[filler])
        c_acc = np.mean(correct_by_filler_condition[filler])
        delta_accuracy_by_filler[filler] = c_acc - b_acc
    
    # A/B fractions
    baseline_a = [1 if r['predicted_choice'] == 'A' else 0 for r in baseline_results['prompt_results']]
    condition_a = [1 if r['predicted_choice'] == 'A' else 0 for r in condition_results['prompt_results']]
    
    return PairedStats(
        condition_name=condition_results['condition_name'],
        baseline_name=baseline_results['condition_name'],
        delta_accuracy=np.mean(delta_correct),
        delta_margin=np.mean(delta_margins),
        delta_accuracy_by_filler=delta_accuracy_by_filler,
        delta_margin_by_filler=delta_margin_by_filler,
        delta_margin_ci_by_filler=delta_margin_ci_by_filler,
        baseline_a_fraction=np.mean(baseline_a),
        condition_a_fraction=np.mean(condition_a),
        n_prompts=len(delta_margins),
    )


def compute_paired_stats_by_order(
    baseline_results: Dict,
    condition_results: Dict,
    order_filter: str,  # 'original' or 'flipped'
) -> Optional[PairedStats]:
    """
    Compute paired statistics filtered by order type (original or flipped).
    """
    # Filter results by order
    baseline_filtered = [r for r in baseline_results['prompt_results'] 
                         if r.get('order', 'original') == order_filter]
    condition_filtered = [r for r in condition_results['prompt_results']
                          if r.get('order', 'original') == order_filter]
    
    if not baseline_filtered or not condition_filtered:
        return None
    
    # Build mock results dicts
    baseline_mock = {'prompt_results': baseline_filtered, 'condition_name': baseline_results['condition_name']}
    condition_mock = {'prompt_results': condition_filtered, 'condition_name': f"{condition_results['condition_name']}_{order_filter}"}
    
    return compute_paired_stats(baseline_mock, condition_mock)


def plot_delta_margin_by_order(
    baseline: Dict,
    ablation: Dict,
    output_file: str,
    token_distances: Dict[int, int] = None,  # filler -> token distance mapping
):
    """
    Plot Δmargin vs token distance with separate curves for original and flipped.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'original': '#2196F3', 'flipped': '#FF5722'}
    markers = {'original': 'o', 'flipped': 's'}
    
    for order in ['original', 'flipped']:
        stats = compute_paired_stats_by_order(baseline, ablation, order)
        if not stats:
            continue
        
        fillers = sorted(stats.delta_margin_by_filler.keys())
        
        # Use token distances if provided, else use filler word counts
        if token_distances:
            x_vals = [token_distances.get(f, f) for f in fillers]
        else:
            x_vals = fillers
        
        means = [stats.delta_margin_by_filler[f] for f in fillers]
        cis = [stats.delta_margin_ci_by_filler[f] for f in fillers]
        
        lower = [ci[0] for ci in cis]
        upper = [ci[1] for ci in cis]
        
        ax.plot(x_vals, means, f'{markers[order]}-', 
                label=f'{order}', color=colors[order], linewidth=2, markersize=8)
        ax.fill_between(x_vals, lower, upper, alpha=0.2, color=colors[order])
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Token Distance', fontsize=12)
    ax.set_ylabel('Δmargin (ablation − baseline)', fontsize=12)
    
    cond_name = ablation['condition_name']
    ax.set_title(f'Δmargin vs Distance: {cond_name} (original vs flipped)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")


def plot_delta_margin_curves(
    paired_stats_list: List[PairedStats],
    output_file: str,
    title: str = "Δmargin vs Distance (Ablation − Baseline)",
):
    """
    Plot Δmargin vs token distance with bootstrap CIs.
    
    This is the "Anthropic circuits blog" style plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(paired_stats_list)))
    
    for stats, color in zip(paired_stats_list, colors):
        fillers = sorted(stats.delta_margin_by_filler.keys())
        means = [stats.delta_margin_by_filler[f] for f in fillers]
        cis = [stats.delta_margin_ci_by_filler[f] for f in fillers]
        
        lower = [ci[0] for ci in cis]
        upper = [ci[1] for ci in cis]
        
        ax.plot(fillers, means, 'o-', label=stats.condition_name, color=color, linewidth=2, markersize=8)
        ax.fill_between(fillers, lower, upper, alpha=0.2, color=color)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Filler Word Count (proxy for token distance)', fontsize=12)
    ax.set_ylabel('Δmargin (ablation − baseline)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")


def print_paired_summary(stats_list: List[PairedStats]):
    """Print summary table of paired statistics."""
    print("\n" + "=" * 80)
    print("PAIRED STATISTICS SUMMARY")
    print("=" * 80)
    
    if not stats_list:
        print("No paired stats to display")
        return
    
    fillers = sorted(stats_list[0].delta_margin_by_filler.keys())
    
    # Header
    print(f"\n{'Condition':<25} {'Δacc':>8} {'Δmargin':>8} {'ΔA%':>8}", end="")
    for filler in fillers:
        print(f" {filler:>6}w", end="")
    print()
    print("-" * 80)
    
    for stats in stats_list:
        delta_a = stats.condition_a_fraction - stats.baseline_a_fraction
        print(f"{stats.condition_name:<25} {stats.delta_accuracy:>+7.1%} {stats.delta_margin:>+7.2f} {delta_a:>+7.1%}", end="")
        for filler in fillers:
            delta = stats.delta_margin_by_filler.get(filler, 0)
            print(f" {delta:>+6.2f}", end="")
        print()
    
    print("=" * 80)
    
    # Print CIs for longest distance
    longest = max(fillers)
    print(f"\n95% Bootstrap CIs for Δmargin at {longest}w:")
    for stats in stats_list:
        ci = stats.delta_margin_ci_by_filler.get(longest, (0, 0))
        mean = stats.delta_margin_by_filler.get(longest, 0)
        print(f"  {stats.condition_name}: {mean:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")


def analyze_symmetry(results_data: Dict) -> Dict[str, Dict]:
    """
    Analyze symmetry control: compare original vs flipped prompts.
    
    Returns stats grouped by order type.
    """
    original_results = []
    flipped_results = []
    
    for r in results_data.get('prompt_results', []):
        order = r.get('order', 'original')
        if order == 'flipped':
            flipped_results.append(r)
        else:
            original_results.append(r)
    
    if not flipped_results:
        return {}
    
    # Compute accuracy for each order
    orig_acc = np.mean([r['is_correct'] for r in original_results])
    flip_acc = np.mean([r['is_correct'] for r in flipped_results])
    
    orig_a_frac = np.mean([1 if r['predicted_choice'] == 'A' else 0 for r in original_results])
    flip_a_frac = np.mean([1 if r['predicted_choice'] == 'A' else 0 for r in flipped_results])
    
    return {
        'original': {'accuracy': orig_acc, 'a_fraction': orig_a_frac, 'n': len(original_results)},
        'flipped': {'accuracy': flip_acc, 'a_fraction': flip_a_frac, 'n': len(flipped_results)},
        'symmetry_delta_accuracy': flip_acc - orig_acc,
        'symmetry_delta_a_fraction': flip_a_frac - orig_a_frac,
    }


def compute_paired_consistency(results_data: Dict) -> Dict[str, any]:
    """
    Compute paired consistency: for each (original, flipped) pair,
    count how often both are correct, neither is correct, or only one is correct.
    
    This tells us if the model is truly retrieving (both correct) or has label bias (mismatch).
    """
    # Build lookup by pair_id
    original_by_id = {}
    flipped_by_pair = {}
    
    for r in results_data.get('prompt_results', []):
        order = r.get('order', 'original')
        if order == 'flipped':
            pair_id = r.get('pair_id')
            if pair_id is not None:
                flipped_by_pair[pair_id] = r
        else:
            original_by_id[r['prompt_id']] = r
    
    if not flipped_by_pair:
        return {}
    
    # Count paired outcomes
    both_correct = 0
    neither_correct = 0
    orig_only = 0  # Original correct, flipped wrong
    flip_only = 0  # Flipped correct, original wrong
    
    for orig_id, orig_r in original_by_id.items():
        if orig_id not in flipped_by_pair:
            continue
        
        flip_r = flipped_by_pair[orig_id]
        o_correct = orig_r['is_correct']
        f_correct = flip_r['is_correct']
        
        if o_correct and f_correct:
            both_correct += 1
        elif not o_correct and not f_correct:
            neither_correct += 1
        elif o_correct and not f_correct:
            orig_only += 1
        else:
            flip_only += 1
    
    n_pairs = both_correct + neither_correct + orig_only + flip_only
    
    return {
        'n_pairs': n_pairs,
        'both_correct': both_correct,
        'neither_correct': neither_correct,
        'orig_only_correct': orig_only,
        'flip_only_correct': flip_only,
        # Percentages
        'pct_both_correct': both_correct / n_pairs * 100 if n_pairs > 0 else 0,
        'pct_neither_correct': neither_correct / n_pairs * 100 if n_pairs > 0 else 0,
        'pct_orig_only': orig_only / n_pairs * 100 if n_pairs > 0 else 0,
        'pct_flip_only': flip_only / n_pairs * 100 if n_pairs > 0 else 0,
        # Consistency = both or neither (model is consistent regardless of label)
        'consistency': (both_correct + neither_correct) / n_pairs * 100 if n_pairs > 0 else 0,
        # Mismatch = only one correct (sign of label bias)
        'mismatch': (orig_only + flip_only) / n_pairs * 100 if n_pairs > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired analysis for intervention experiments")
    parser.add_argument('--results_dir', type=str, default='intervention_results',
                        help='Directory containing results JSON files')
    parser.add_argument('--output_dir', type=str, default='intervention_outputs',
                        help='Directory for output plots')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Specific results file to analyze (if not using --results_dir)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Comma-separated condition names to include in plot (partial match)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find results files
    if args.results_file:
        results_files = [Path(args.results_file)]
    else:
        results_dir = Path(args.results_dir)
        results_files = sorted(results_dir.glob('*.json'))
    
    if not results_files:
        print("No results files found!")
        return
    
    # Load and merge all conditions from all files
    all_conditions = []
    for rf in results_files:
        try:
            data = load_results(str(rf))
            conditions = data.get('conditions', [])
            n_cond = len(conditions)
            cond_names = [c.get('condition_name', 'unknown') for c in conditions]
            
            # Skip baseline-only files
            if n_cond == 1 and cond_names[0] == 'baseline':
                print(f"  Skipping (baseline only): {rf.name}")
                continue
            
            print(f"  Loaded {rf.name}: {n_cond} conditions ({', '.join(cond_names)})")
            all_conditions.extend(conditions)
        except Exception as e:
            print(f"  Error loading {rf}: {e}")
    
    if not all_conditions:
        print("No valid conditions found in any file!")
        return
    
    # Find baseline
    baseline = None
    ablations = []
    for c in all_conditions: # Changed from 'conditions' to 'all_conditions'
        if c['condition_name'] == 'baseline':
            baseline = c
        else:
            ablations.append(c)
    
    if not baseline:
        print("No baseline condition found!")
        return
    
    # Compute paired stats
    paired_stats = []
    for ablation in ablations:
        stats = compute_paired_stats(baseline, ablation)
        paired_stats.append(stats)
    
    # Print summary
    print_paired_summary(paired_stats)
    
    # Check for symmetry data
    for c in all_conditions:
        sym_stats = analyze_symmetry(c)
        if sym_stats:
            print(f"\n--- Symmetry Analysis for {c['condition_name']} ---")
            print(f"  Original: acc={sym_stats['original']['accuracy']:.1%}, A%={sym_stats['original']['a_fraction']:.1%}")
            print(f"  Flipped:  acc={sym_stats['flipped']['accuracy']:.1%}, A%={sym_stats['flipped']['a_fraction']:.1%}")
            print(f"  Δacc (flipped-original): {sym_stats['symmetry_delta_accuracy']:+.1%}")
            print(f"  ΔA% (flipped-original): {sym_stats['symmetry_delta_a_fraction']:+.1%}")
    
    # Paired consistency analysis (for original+flipped data)
    print("\n" + "=" * 80)
    print("PAIRED CONSISTENCY ANALYSIS (original↔flipped pairs)")
    print("=" * 80)
    print(f"{'Condition':<25} {'Both✓':>8} {'Neither':>8} {'Orig✓':>8} {'Flip✓':>8} {'Consist':>8} {'Mismatch':>8}")
    print("-" * 80)
    
    for c in all_conditions:
        pair_stats = compute_paired_consistency(c)
        if pair_stats:
            print(f"{c['condition_name']:<25} "
                  f"{pair_stats['pct_both_correct']:>7.1f}% "
                  f"{pair_stats['pct_neither_correct']:>7.1f}% "
                  f"{pair_stats['pct_orig_only']:>7.1f}% "
                  f"{pair_stats['pct_flip_only']:>7.1f}% "
                  f"{pair_stats['consistency']:>7.1f}% "
                  f"{pair_stats['mismatch']:>7.1f}%")
    
    print("-" * 80)
    print("Both✓ = both correct, Neither = both wrong, Orig✓/Flip✓ = only one correct")
    print("Consistency = Both✓ + Neither (model ignores label), Mismatch = Orig✓ + Flip✓ (label bias)")
    print("=" * 80)
    
    # Combined table: Δaccuracy per bin + paired consistency
    print("\n" + "=" * 100)
    print("COMBINED ANALYSIS: Δaccuracy by distance + paired consistency")
    print("=" * 100)
    
    if baseline and len(ablations) > 0:
        # Get filler bins from first ablation
        sample_stats = compute_paired_stats(baseline, ablations[0])
        fillers = sorted(sample_stats.delta_accuracy_by_filler.keys())
        
        # Header
        header = f"{'Condition':<20}"
        for f in fillers:
            header += f" {f:>5}w"
        header += f" {'Both✓':>7} {'Neither':>7} {'Mismatch':>8}"
        print(header)
        print("-" * 100)
        
        for ablation in ablations:
            stats = compute_paired_stats(baseline, ablation)
            pair_stats = compute_paired_consistency(ablation)
            
            row = f"{ablation['condition_name']:<20}"
            for f in fillers:
                delta = stats.delta_accuracy_by_filler.get(f, 0)
                row += f" {delta:>+5.1%}"
            
            if pair_stats:
                row += f" {pair_stats['pct_both_correct']:>6.1f}%"
                row += f" {pair_stats['pct_neither_correct']:>6.1f}%"
                row += f" {pair_stats['mismatch']:>7.1f}%"
            
            print(row)
        
        print("=" * 100)
    
    # Generate original vs flipped plot for each ablation condition
    if baseline:
        # Token distance mapping (from filler word count to actual tokens)
        # These are typical values - adjust based on actual data
        token_distances = {0: 34, 100: 160, 300: 412, 600: 731}
        
        for ablation in ablations:
            cond_name = ablation['condition_name'].replace('+', '').replace('L23_', '')
            plot_file = output_dir / f"delta_margin_orig_vs_flip_{cond_name}.png"
            plot_delta_margin_by_order(baseline, ablation, str(plot_file), token_distances)
    
    # Generate standard plot (filtered if requested)
    if paired_stats:
        # Filter conditions if requested
        if args.filter:
            filter_patterns = [p.strip() for p in args.filter.split(',')]
            filtered_stats = []
            for stats in paired_stats:
                cond = stats.condition_name
                # Match: exact name, name with L23_ prefix, or short form without prefix
                name_variants = [
                    cond,
                    cond.replace('L23_', ''),
                    f"L23_{cond}" if not cond.startswith('L23') else cond,
                ]
                for pattern in filter_patterns:
                    pattern_variants = [pattern, f"L23_{pattern}", pattern.replace('L23_', '')]
                    if any(v in name_variants for v in pattern_variants):
                        filtered_stats.append(stats)
                        break
            print(f"\nFiltered to {len(filtered_stats)} conditions: {[s.condition_name for s in filtered_stats]}")
            paired_stats = filtered_stats
        
        if paired_stats:
            cond_names = '_'.join(s.condition_name.replace('+', '').replace('L23_', '') for s in paired_stats[:4])
            plot_file = output_dir / f"delta_margin_{cond_names}.png"
            plot_delta_margin_curves(paired_stats, str(plot_file))
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
