"""
Analyze Intervention Experiment Results.

Processes results from run_intervention.py and generates:
- Accuracy tables by condition × filler length
- Logprob margin distributions
- Statistical significance tests
- Visualization plots

Usage:
    python analyze_intervention.py --results_dir ./intervention_results
    python analyze_intervention.py --results_file results_20260103.json
"""

import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math

# Optional: matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

# Optional: scipy for statistical tests
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, skipping statistical tests")


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class PromptResult:
    """Reconstructed prompt result from JSON."""
    prompt_id: int
    filler_word_count: int
    correct_choice: str
    correct_value: int
    generated_text: str
    predicted_choice: str
    is_correct: bool
    logprobs: Dict[str, float]
    margin: Optional[float]


@dataclass
class ConditionData:
    """Condition data loaded from JSON."""
    condition_name: str
    ablated_heads: List[Tuple[int, int]]
    overall_accuracy: float
    accuracy_by_filler: Dict[int, float]
    mean_margin: float
    margin_by_filler: Dict[int, float]
    n_prompts: int
    prompt_results: List[PromptResult]


def load_results(filepath: str) -> List[ConditionData]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    conditions = []
    for cond in data['conditions']:
        # Parse prompt results
        prompt_results = []
        for pr in cond['prompt_results']:
            prompt_results.append(PromptResult(
                prompt_id=pr['prompt_id'],
                filler_word_count=pr['filler_word_count'],
                correct_choice=pr['correct_choice'],
                correct_value=pr['correct_value'],
                generated_text=pr['generated_text'],
                predicted_choice=pr['predicted_choice'],
                is_correct=pr['is_correct'],
                logprobs=pr.get('logprobs', {}),
                margin=pr.get('margin'),
            ))
        
        # Convert string keys back to int
        accuracy_by_filler = {int(k): v for k, v in cond['accuracy_by_filler'].items()}
        margin_by_filler = {int(k): v for k, v in cond.get('margin_by_filler', {}).items()}
        
        conditions.append(ConditionData(
            condition_name=cond['condition_name'],
            ablated_heads=[tuple(h) for h in cond['ablated_heads']],
            overall_accuracy=cond['overall_accuracy'],
            accuracy_by_filler=accuracy_by_filler,
            mean_margin=cond['mean_margin'],
            margin_by_filler=margin_by_filler,
            n_prompts=cond['n_prompts'],
            prompt_results=prompt_results,
        ))
    
    return conditions


def find_latest_results(results_dir: str) -> str:
    """Find the most recent results file in the directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    result_files = list(results_path.glob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    # Sort by modification time
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(result_files[0])


# =============================================================================
# Analysis
# =============================================================================

def compute_accuracy_delta(
    baseline: ConditionData,
    ablated: ConditionData,
) -> Dict[int, float]:
    """
    Compute accuracy delta (baseline - ablated) by filler length.
    
    Positive delta means ablation hurt performance.
    """
    deltas = {}
    for filler in baseline.accuracy_by_filler:
        if filler in ablated.accuracy_by_filler:
            deltas[filler] = baseline.accuracy_by_filler[filler] - ablated.accuracy_by_filler[filler]
    return deltas


def compute_margin_delta(
    baseline: ConditionData,
    ablated: ConditionData,
) -> Dict[int, float]:
    """
    Compute margin delta (baseline - ablated) by filler length.
    
    Positive delta means ablation reduced confidence margin.
    """
    deltas = {}
    for filler in baseline.margin_by_filler:
        if filler in ablated.margin_by_filler:
            deltas[filler] = baseline.margin_by_filler[filler] - ablated.margin_by_filler[filler]
    return deltas


def paired_t_test(
    baseline: ConditionData,
    ablated: ConditionData,
    metric: str = 'margin',  # 'margin' or 'is_correct'
) -> Dict[int, Tuple[float, float]]:
    """
    Run paired t-test comparing baseline vs ablated for each filler length.
    
    Returns dict mapping filler_length to (t_statistic, p_value).
    """
    if not HAS_SCIPY:
        return {}
    
    # Group by filler length
    baseline_by_filler: Dict[int, List] = {}
    ablated_by_filler: Dict[int, List] = {}
    
    for pr in baseline.prompt_results:
        if pr.filler_word_count not in baseline_by_filler:
            baseline_by_filler[pr.filler_word_count] = []
        val = pr.margin if metric == 'margin' else int(pr.is_correct)
        if val is not None:
            baseline_by_filler[pr.filler_word_count].append((pr.prompt_id, val))
    
    for pr in ablated.prompt_results:
        if pr.filler_word_count not in ablated_by_filler:
            ablated_by_filler[pr.filler_word_count] = []
        val = pr.margin if metric == 'margin' else int(pr.is_correct)
        if val is not None:
            ablated_by_filler[pr.filler_word_count].append((pr.prompt_id, val))
    
    results = {}
    for filler in baseline_by_filler:
        if filler not in ablated_by_filler:
            continue
        
        # Match by prompt_id
        baseline_dict = dict(baseline_by_filler[filler])
        ablated_dict = dict(ablated_by_filler[filler])
        
        common_ids = set(baseline_dict.keys()) & set(ablated_dict.keys())
        if len(common_ids) < 2:
            continue
        
        baseline_vals = [baseline_dict[pid] for pid in sorted(common_ids)]
        ablated_vals = [ablated_dict[pid] for pid in sorted(common_ids)]
        
        t_stat, p_val = stats.ttest_rel(baseline_vals, ablated_vals)
        results[filler] = (t_stat, p_val)
    
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_accuracy_table(conditions: List[ConditionData]):
    """Print accuracy table by condition × filler length."""
    
    print("\n" + "=" * 80)
    print("ACCURACY BY CONDITION AND FILLER LENGTH")
    print("=" * 80)
    
    # Get all filler lengths
    all_fillers = set()
    for cond in conditions:
        all_fillers.update(cond.accuracy_by_filler.keys())
    filler_lengths = sorted(all_fillers)
    
    # Header
    header = f"{'Condition':<30} {'Overall':>8}"
    for filler in filler_lengths:
        header += f" {filler:>6}w"
    print(header)
    print("-" * 80)
    
    # Rows
    for cond in conditions:
        row = f"{cond.condition_name:<30} {cond.overall_accuracy:>7.1%}"
        for filler in filler_lengths:
            acc = cond.accuracy_by_filler.get(filler, float('nan'))
            if math.isnan(acc):
                row += f" {'N/A':>6}"
            else:
                row += f" {acc:>6.1%}"
        print(row)
    
    print("=" * 80)


def print_margin_table(conditions: List[ConditionData]):
    """Print margin table by condition × filler length."""
    
    print("\n" + "=" * 80)
    print("LOGPROB MARGIN BY CONDITION AND FILLER LENGTH")
    print("=" * 80)
    
    # Get all filler lengths
    all_fillers = set()
    for cond in conditions:
        all_fillers.update(cond.margin_by_filler.keys())
    filler_lengths = sorted(all_fillers)
    
    # Header
    header = f"{'Condition':<30} {'Mean':>8}"
    for filler in filler_lengths:
        header += f" {filler:>7}w"
    print(header)
    print("-" * 80)
    
    # Rows
    for cond in conditions:
        row = f"{cond.condition_name:<30} {cond.mean_margin:>7.2f}"
        for filler in filler_lengths:
            margin = cond.margin_by_filler.get(filler, float('nan'))
            if math.isnan(margin):
                row += f" {'N/A':>7}"
            else:
                row += f" {margin:>7.2f}"
        print(row)
    
    print("=" * 80)


def print_delta_analysis(conditions: List[ConditionData]):
    """Print delta analysis comparing baseline to each ablation condition."""
    
    # Find baseline
    baseline = None
    for cond in conditions:
        if 'baseline' in cond.condition_name.lower():
            baseline = cond
            break
    
    if baseline is None:
        print("No baseline condition found for delta analysis")
        return
    
    print("\n" + "=" * 80)
    print("DELTA ANALYSIS (Baseline - Ablated)")
    print("Positive delta = ablation hurt performance")
    print("=" * 80)
    
    filler_lengths = sorted(baseline.accuracy_by_filler.keys())
    
    for cond in conditions:
        if cond.condition_name == baseline.condition_name:
            continue
        
        print(f"\n--- {cond.condition_name} ---")
        
        # Accuracy deltas
        acc_deltas = compute_accuracy_delta(baseline, cond)
        print(f"  Accuracy delta by filler:")
        for filler in filler_lengths:
            if filler in acc_deltas:
                delta = acc_deltas[filler]
                sign = "+" if delta > 0 else ""
                print(f"    {filler}w: {sign}{delta:.1%}")
        
        # Margin deltas
        margin_deltas = compute_margin_delta(baseline, cond)
        print(f"  Margin delta by filler:")
        for filler in filler_lengths:
            if filler in margin_deltas:
                delta = margin_deltas[filler]
                sign = "+" if delta > 0 else ""
                print(f"    {filler}w: {sign}{delta:.2f}")
        
        # Statistical significance
        if HAS_SCIPY:
            t_tests = paired_t_test(baseline, cond, metric='margin')
            print(f"  Paired t-test (margin):")
            for filler in filler_lengths:
                if filler in t_tests:
                    t_stat, p_val = t_tests[filler]
                    sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"    {filler}w: t={t_stat:.2f}, p={p_val:.4f} {sig}")


def print_interpretation(conditions: List[ConditionData]):
    """Print interpretation of results."""
    
    # Find baseline and target ablation
    baseline = None
    target_ablation = None
    
    for cond in conditions:
        if 'baseline' in cond.condition_name.lower():
            baseline = cond
        elif 'ablate_L23H1' in cond.condition_name or 'ablate' in cond.condition_name.lower():
            target_ablation = cond
    
    if baseline is None or target_ablation is None:
        return
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Find filler length with biggest margin collapse
    margin_deltas = compute_margin_delta(baseline, target_ablation)
    if margin_deltas:
        max_delta_filler = max(margin_deltas, key=margin_deltas.get)
        max_delta = margin_deltas[max_delta_filler]
        
        # Check short vs long distance pattern
        short_fillers = [f for f in margin_deltas if f <= 100]
        long_fillers = [f for f in margin_deltas if f >= 400]
        
        short_avg = sum(margin_deltas[f] for f in short_fillers) / len(short_fillers) if short_fillers else 0
        long_avg = sum(margin_deltas[f] for f in long_fillers) / len(long_fillers) if long_fillers else 0
        
        print(f"\nTarget ablation: {target_ablation.condition_name}")
        print(f"  Maximum margin collapse: {max_delta:.2f} at {max_delta_filler}w filler")
        print(f"  Short distance (≤100w) avg delta: {short_avg:.2f}")
        print(f"  Long distance (≥400w) avg delta: {long_avg:.2f}")
        
        if long_avg > short_avg + 0.5:
            print("\n  → Pattern matches hypothesis: ablation hurts long-distance more than short-distance")
        elif long_avg > short_avg:
            print("\n  → Weak support for hypothesis: slight long-distance effect")
        else:
            print("\n  → Pattern does NOT match hypothesis: no distance-dependent effect")


# =============================================================================
# Plotting
# =============================================================================

def plot_accuracy_by_filler(conditions: List[ConditionData], output_path: str = None):
    """Plot accuracy by filler length for all conditions."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all filler lengths
    all_fillers = set()
    for cond in conditions:
        all_fillers.update(cond.accuracy_by_filler.keys())
    filler_lengths = sorted(all_fillers)
    
    # Plot each condition
    for cond in conditions:
        accuracies = [cond.accuracy_by_filler.get(f, float('nan')) for f in filler_lengths]
        label = cond.condition_name.replace('_', ' ')
        ax.plot(filler_lengths, accuracies, marker='o', label=label)
    
    ax.set_xlabel('Filler Length (words)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Filler Length and Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_margin_by_filler(conditions: List[ConditionData], output_path: str = None):
    """Plot logprob margin by filler length for all conditions."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all filler lengths
    all_fillers = set()
    for cond in conditions:
        all_fillers.update(cond.margin_by_filler.keys())
    filler_lengths = sorted(all_fillers)
    
    # Plot each condition
    for cond in conditions:
        margins = [cond.margin_by_filler.get(f, float('nan')) for f in filler_lengths]
        label = cond.condition_name.replace('_', ' ')
        ax.plot(filler_lengths, margins, marker='o', label=label)
    
    ax.set_xlabel('Filler Length (words)')
    ax.set_ylabel('Logprob Margin (correct - best wrong)')
    ax.set_title('Confidence Margin by Filler Length and Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved margin plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze intervention results")
    
    parser.add_argument('--results_dir', type=str, default='./intervention_results',
                        help='Directory containing results files')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Specific results file to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output plots (default: same as results)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Find results file
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = find_latest_results(args.results_dir)
    
    print(f"Analyzing: {results_file}")
    
    # Load results
    conditions = load_results(results_file)
    print(f"Loaded {len(conditions)} conditions")
    
    # Print tables
    print_accuracy_table(conditions)
    print_margin_table(conditions)
    print_delta_analysis(conditions)
    print_interpretation(conditions)
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir = args.output_dir or args.results_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        plot_accuracy_by_filler(
            conditions, 
            output_path=str(Path(output_dir) / "accuracy_by_filler.png")
        )
        plot_margin_by_filler(
            conditions,
            output_path=str(Path(output_dir) / "margin_by_filler.png")
        )


if __name__ == "__main__":
    main()
