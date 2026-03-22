"""
Intervention Prompts Generator v2 for Gemma-2 Ablation Experiments.

CRITICAL FIX: Filler is BETWEEN table and query, and token_distance is computed
as the actual token gap from end-of-table to Answer position.

Changes from v1:
- Fixed 16 rows per table (not variable)
- 6-digit values with high separation
- Filler text stripped of all digits
- Token distance computed properly via tokenizer
- Prefix/middle/suffix structure for clean distance computation

Usage:
    python intervention_prompts.py --n_prompts 100 --output prompts.json
"""

import json
import random
import string
import re
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from pathlib import Path


# =============================================================================
# Default Wikipedia Filler Text (stripped of digits)
# =============================================================================

DEFAULT_FILLER_TEXT = """
A forest is an area of land dominated by trees. Hundreds of definitions of forest are used throughout the world, incorporating factors such as tree density, tree height, land use, legal standing, and ecological function. The United Nations Food and Agriculture Organization defines a forest as land spanning more than half a hectare with trees higher than five meters and a canopy cover of more than ten percent, or trees able to reach these thresholds in situ.

Forests are the dominant terrestrial ecosystem of Earth, and are distributed around the globe. More than half of the world's forests are found in only five countries: Brazil, Canada, China, Russia, and the United States. The largest share of forests is in tropical regions, followed by boreal, temperate, and subtropical domains.

Forests account for seventy-five percent of the gross primary production of the Earth's biosphere, and contain eighty percent of the Earth's plant biomass. Net primary production is estimated at twenty-two gigatonnes of biomass per year for tropical forests, eight for temperate forests, and three for boreal forests.

Forests at different latitudes and elevations, and with different precipitation and evapotranspiration form distinctly different biomes: boreal forests around the poles, tropical moist forests and tropical dry forests around the Equator, and temperate forests at the middle latitudes. Higher-elevation areas tend to support forests similar to those at higher latitudes, and the amount of precipitation also affects forest composition.

Almost half the forest area is relatively intact, and more than one-third is primary forest. More than half of the world's forests is found in only five countries. The world's total forest area is declining at an accelerating rate, with the primary drivers of deforestation including agriculture, logging, and wildfires.

Human society and forests influence each other in both positive and negative ways. Forests provide ecosystem services to humans and serve as tourist attractions. Forests can also affect people's health. Human activities, including unsustainable use of forest resources, can negatively affect forest ecosystems.

Forest management has evolved from a focus on wood extraction to include maintaining biodiversity and supporting ecosystem services. National and international policies and programs are aimed at forest conservation and sustainable use.

The understory of a forest consists of trees that are shorter than the canopy layer. It receives less sunlight than the canopy, producing a distinctive environment. The forest floor is the lowest layer where decomposition takes place.

Trees in the forest release oxygen through photosynthesis and absorb carbon dioxide from the atmosphere. This process helps regulate the Earth's climate. Forests also play a crucial role in the water cycle by releasing water vapor into the atmosphere.

Many species of animals make their homes in forests. These include mammals, birds, insects, and amphibians. The diversity of life in forests is a result of the variety of habitats that forests provide.

Temperate deciduous forests experience all four seasons. Trees in these forests lose their leaves in autumn and grow new ones in spring. Common trees include oak, maple, and beech.

Tropical rainforests are found near the equator. They receive large amounts of rainfall throughout the year. These forests are home to more species of plants and animals than any other ecosystem.

Boreal forests are found in cold climates near the Arctic. They are dominated by coniferous trees such as spruce, pine, and fir. These forests experience long, cold winters and short, cool summers.
""".strip()


def strip_digits(text: str) -> str:
    """Remove all digits from text to prevent numeric interference."""
    return re.sub(r'\d+', '', text)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    
    # Table configuration - FIXED values for cleaner experiment
    n_entries: int = 16  # Fixed 16 rows
    value_digits: int = 6  # 6-digit values for high separation
    value_min_separation: int = 10000  # Min difference between values
    
    # ID format: 1 uppercase letter + 2 digits
    id_prefix_length: int = 1
    id_suffix_digits: int = 2
    
    # Filler configuration (word counts = distance bins)
    filler_word_counts: List[int] = field(
        default_factory=lambda: [0, 100, 300, 600]
    )
    filler_text_file: str = "filler_text_clean.txt"
    
    # Choice configuration
    n_choices: int = 2  # A/B only
    
    # Output
    n_prompts_per_filler_length: int = 100
    output_file: str = "intervention_prompts.json"
    
    # Random seed
    seed: int = 42


@dataclass  
class GeneratedPrompt:
    """A single generated prompt with metadata."""
    
    prompt_id: int
    prompt_text: str
    
    # Ground truth
    query_id: str
    correct_value: int
    correct_choice: str
    
    # Metadata
    n_entries: int
    filler_word_count: int
    n_choices: int
    
    # Symmetry control: 'original' or 'flipped' (A↔B swapped)
    order: str = "original"  # 'original' or 'flipped'
    pair_id: Optional[int] = None  # ID of paired prompt (for matching original↔flipped)
    
    # Distance info (computed via tokenizer later)
    token_distance: Optional[int] = None
    
    # Prompt parts for distance computation
    prefix: str = ""  # header + table
    middle: str = ""  # filler
    suffix: str = ""  # query + choices + Answer:
    
    # All choices and table for reference
    choices: List[Tuple[str, int]] = field(default_factory=list)
    table: List[Tuple[str, int]] = field(default_factory=list)


# =============================================================================
# ID/Value Generation
# =============================================================================

def generate_distinct_id(used_ids: set, prefix_len: int = 1, suffix_digits: int = 2) -> str:
    """Generate a distinct ID that doesn't share first letter with existing IDs."""
    max_attempts = 1000
    
    for _ in range(max_attempts):
        prefix = ''.join(random.choices(string.ascii_uppercase, k=prefix_len))
        suffix = str(random.randint(10 ** (suffix_digits - 1), 10 ** suffix_digits - 1))
        candidate = prefix + suffix
        
        if candidate not in used_ids:
            first_letters = {id_[0] for id_ in used_ids}
            if prefix[0] not in first_letters:
                return candidate
    
    # Fallback
    for _ in range(max_attempts):
        prefix = ''.join(random.choices(string.ascii_uppercase, k=prefix_len))
        suffix = str(random.randint(10, 99))
        candidate = prefix + suffix
        if candidate not in used_ids:
            return candidate
    
    raise ValueError("Could not generate distinct ID")


def generate_separated_values(n: int, n_digits: int = 6, min_sep: int = 10000) -> List[int]:
    """Generate n values with guaranteed minimum separation."""
    min_val = 10 ** (n_digits - 1)
    max_val = 10 ** n_digits - 1
    
    # Use spaced grid to ensure separation
    range_size = max_val - min_val
    step = range_size // (n + 1)
    
    values = []
    for i in range(1, n + 1):
        base = min_val + i * step
        # Add small random jitter
        jitter = random.randint(-step // 4, step // 4)
        val = base + jitter
        val = max(min_val, min(max_val, val))
        values.append(val)
    
    random.shuffle(values)
    return values


def generate_table(n_entries: int, config: PromptConfig) -> List[Tuple[str, int]]:
    """Generate a table of distinct ID→VALUE pairs with separated values."""
    used_ids = set()
    table = []
    
    # Generate separated values
    values = generate_separated_values(n_entries, config.value_digits, config.value_min_separation)
    
    for i in range(n_entries):
        id_ = generate_distinct_id(
            used_ids, 
            prefix_len=config.id_prefix_length,
            suffix_digits=config.id_suffix_digits
        )
        used_ids.add(id_)
        table.append((id_, values[i]))
    
    return table


# =============================================================================
# Filler Text Management
# =============================================================================

def load_or_create_filler(config: PromptConfig) -> str:
    """Load filler text from cache or create from default (digits stripped)."""
    filler_path = Path(config.filler_text_file)
    
    if filler_path.exists():
        text = filler_path.read_text(encoding='utf-8')
        return strip_digits(text)
    
    # Create cache file with digit-free text
    clean_text = strip_digits(DEFAULT_FILLER_TEXT)
    filler_path.write_text(clean_text, encoding='utf-8')
    print(f"Created digit-free filler cache: {filler_path}")
    
    return clean_text


def get_filler_segment(filler_text: str, word_count: int) -> str:
    """Extract a segment of filler text with approximately word_count words."""
    if word_count == 0:
        return ""
    
    words = filler_text.split()
    if word_count >= len(words):
        return filler_text
    
    return ' '.join(words[:word_count])


# =============================================================================
# Prompt Construction - NEW STRUCTURE
# =============================================================================

# Prompt is built in 3 parts: prefix (header+table), middle (filler), suffix (query+answer)
# This makes token distance computation clean and correct.

HEADER = """Task: Choose the correct VALUE for the Query ID.
Rules: Output ONLY A or B. Output nothing else.

DATA:
"""

def format_table(table: List[Tuple[str, int]]) -> str:
    """Format table as ID: X VALUE: Y lines, one per line."""
    lines = [f"ID: {id_} VALUE: {value}" for id_, value in table]
    return '\n'.join(lines)


def format_choices(choices: List[Tuple[str, int]]) -> str:
    """Format choices, one per line."""
    return '\n'.join([f"{letter}) {value}" for letter, value in choices])


def generate_choices(
    table: List[Tuple[str, int]], 
    target_idx: int, 
    n_choices: int,
    force_correct_position: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], str]:
    """Generate answer choices with balanced positions."""
    choice_letters = ['A', 'B', 'C', 'D'][:n_choices]
    target_id, target_value = table[target_idx]
    
    # Get distractor from a different row (not adjacent to reduce confusion)
    other_indices = [i for i in range(len(table)) if abs(i - target_idx) > 2]
    if not other_indices:
        other_indices = [i for i in range(len(table)) if i != target_idx]
    
    distractor_idx = random.choice(other_indices)
    distractor_value = table[distractor_idx][1]
    
    if force_correct_position is not None:
        if force_correct_position == 0:
            all_values = [target_value, distractor_value]
        else:
            all_values = [distractor_value, target_value]
        correct_position = force_correct_position
    else:
        all_values = [target_value, distractor_value]
        random.shuffle(all_values)
        correct_position = all_values.index(target_value)
    
    correct_letter = choice_letters[correct_position]
    choices = list(zip(choice_letters, all_values))
    
    return choices, correct_letter


def build_prompt_parts(
    table: List[Tuple[str, int]],
    query_id: str,
    choices: List[Tuple[str, int]],
    filler_segment: str,
) -> Tuple[str, str, str]:
    """
    Build prompt in 3 parts for clean distance computation:
    - prefix: header + table (ends with newlines)
    - middle: filler text (what creates the distance)
    - suffix: query + choices + "Answer: "
    """
    # Permute table rows to avoid position bias
    table_permuted = table[:]
    random.shuffle(table_permuted)
    
    table_text = format_table(table_permuted)
    choices_text = format_choices(choices)
    
    # Prefix: everything up to and including the table
    prefix = HEADER + table_text + "\n\n"
    
    # Middle: filler (this is what creates the token distance)
    if filler_segment:
        middle = f"[FILLER START]\n{filler_segment}\n[FILLER END]\n\n"
    else:
        middle = ""
    
    # Suffix: query through Answer:
    suffix = f"Query: ID={query_id}\n\nChoices:\n{choices_text}\n\nAnswer: "
    
    return prefix, middle, suffix


def generate_single_prompt(
    prompt_id: int,
    table: List[Tuple[str, int]],
    filler_segment: str,
    filler_word_count: int,
    n_choices: int,
    force_correct_position: Optional[int] = None,
    order: str = "original",
    pair_id: Optional[int] = None,
) -> GeneratedPrompt:
    """Generate a single prompt with proper part separation."""
    
    # Select random query target
    target_idx = random.randint(0, len(table) - 1)
    query_id, correct_value = table[target_idx]
    
    # Generate choices
    choices, correct_letter = generate_choices(table, target_idx, n_choices, force_correct_position)
    
    # Build prompt in parts
    prefix, middle, suffix = build_prompt_parts(table, query_id, choices, filler_segment)
    
    # Full prompt
    prompt_text = prefix + middle + suffix
    
    return GeneratedPrompt(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        query_id=query_id,
        correct_value=correct_value,
        correct_choice=correct_letter,
        n_entries=len(table),
        filler_word_count=filler_word_count,
        n_choices=n_choices,
        order=order,
        pair_id=pair_id,
        token_distance=None,  # Computed later with tokenizer
        prefix=prefix,
        middle=middle,
        suffix=suffix,
        choices=choices,
        table=table,
    )


# =============================================================================
# Token Distance Computation
# =============================================================================

def compute_token_distances(prompts: List[GeneratedPrompt], tokenizer) -> List[GeneratedPrompt]:
    """
    Compute actual token distances for each prompt using the tokenizer.
    
    token_distance = tokens from end-of-prefix (end of table) to end of prompt (Answer:)
    """
    for prompt in prompts:
        # Tokenize prefix (header + table)
        prefix_tokens = tokenizer.encode(prompt.prefix, add_special_tokens=False)
        
        # Tokenize full prompt
        full_tokens = tokenizer.encode(prompt.prompt_text, add_special_tokens=False)
        
        # Distance = full length - prefix length
        # This is how many tokens of "middle + suffix" there are
        prompt.token_distance = len(full_tokens) - len(prefix_tokens)
    
    return prompts


# =============================================================================
# Main Generator
# =============================================================================

def flip_prompt(prompt: GeneratedPrompt, new_prompt_id: int) -> GeneratedPrompt:
    """
    Create a flipped version of a prompt where A↔B are swapped.
    
    This is the symmetry control: if the model is actually retrieving,
    performance should be invariant to this swap.
    """
    # Swap the choices (A becomes B's value, B becomes A's value)
    if len(prompt.choices) != 2:
        raise ValueError("flip_prompt only works with 2 choices")
    
    orig_a_value = prompt.choices[0][1]  # Value that was A
    orig_b_value = prompt.choices[1][1]  # Value that was B
    
    # Flipped choices: A now has B's value, B now has A's value
    flipped_choices = [('A', orig_b_value), ('B', orig_a_value)]
    
    # Flip the correct choice
    flipped_correct = 'B' if prompt.correct_choice == 'A' else 'A'
    
    # Build new suffix with flipped choices
    choices_text = format_choices(flipped_choices)
    new_suffix = f"Query: ID={prompt.query_id}\n\nChoices:\n{choices_text}\n\nAnswer: "
    
    # Full prompt with flipped suffix
    new_prompt_text = prompt.prefix + prompt.middle + new_suffix
    
    return GeneratedPrompt(
        prompt_id=new_prompt_id,
        prompt_text=new_prompt_text,
        query_id=prompt.query_id,
        correct_value=prompt.correct_value,
        correct_choice=flipped_correct,
        n_entries=prompt.n_entries,
        filler_word_count=prompt.filler_word_count,
        n_choices=prompt.n_choices,
        order="flipped",
        pair_id=prompt.prompt_id,  # Link back to original
        token_distance=prompt.token_distance,
        prefix=prompt.prefix,
        middle=prompt.middle,
        suffix=new_suffix,
        choices=flipped_choices,
        table=prompt.table,
    )


def generate_prompt_set(config: PromptConfig, include_flipped: bool = False) -> List[GeneratedPrompt]:
    """
    Generate full set of prompts across all filler lengths.
    Enforces 50/50 A/B balance within each filler bin.
    
    If include_flipped=True, also generates A↔B swapped copies for symmetry control.
    """
    random.seed(config.seed)
    
    # Load filler text
    filler_text = load_or_create_filler(config)
    
    prompts = []
    prompt_id = 0
    
    for filler_words in config.filler_word_counts:
        filler_segment = get_filler_segment(filler_text, filler_words)
        
        # Enforce 50/50 balance
        n_prompts = config.n_prompts_per_filler_length
        positions = [0] * (n_prompts // 2) + [1] * (n_prompts - n_prompts // 2)
        random.shuffle(positions)
        
        for i in range(n_prompts):
            # Generate table with fixed row count
            table = generate_table(config.n_entries, config)
            
            force_pos = positions[i] if config.n_choices == 2 else None
            
            prompt = generate_single_prompt(
                prompt_id=prompt_id,
                table=table,
                filler_segment=filler_segment,
                filler_word_count=filler_words,
                n_choices=config.n_choices,
                force_correct_position=force_pos,
                order="original",
                pair_id=None,
            )
            
            prompts.append(prompt)
            prompt_id += 1
            
            # Create flipped version if requested
            if include_flipped and config.n_choices == 2:
                flipped = flip_prompt(prompt, prompt_id)
                prompts.append(flipped)
                prompt_id += 1
    
    return prompts


def save_prompts(prompts: List[GeneratedPrompt], output_file: str):
    """Save prompts to JSON file."""
    data = {
        'n_prompts': len(prompts),
        'filler_word_counts': sorted(set(p.filler_word_count for p in prompts)),
        'prompts': [asdict(p) for p in prompts],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(prompts)} prompts to {output_file}")


def load_prompts(input_file: str) -> List[GeneratedPrompt]:
    """Load prompts from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for p in data['prompts']:
        # Convert lists back to tuples
        p['choices'] = [tuple(c) for c in p['choices']]
        p['table'] = [tuple(t) for t in p['table']]
        prompts.append(GeneratedPrompt(**p))
    
    return prompts


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate intervention prompts v2")
    parser.add_argument('--n_prompts', type=int, default=100,
                        help='Number of prompts per filler length')
    parser.add_argument('--n_choices', type=int, default=2, choices=[2, 4],
                        help='Number of answer choices')
    parser.add_argument('--output', type=str, default='intervention_prompts.json',
                        help='Output JSON file')
    parser.add_argument('--filler_lengths', type=str, default='0,100,300,600',
                        help='Comma-separated filler word counts')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--preview', action='store_true',
                        help='Preview first prompt of each filler length')
    parser.add_argument('--include_flipped', action='store_true',
                        help='Include A↔B swapped copies for symmetry control')
    
    args = parser.parse_args()
    
    filler_lengths = [int(x) for x in args.filler_lengths.split(',')]
    
    config = PromptConfig(
        n_prompts_per_filler_length=args.n_prompts,
        n_choices=args.n_choices,
        output_file=args.output,
        filler_word_counts=filler_lengths,
        seed=args.seed,
    )
    
    print(f"Generating prompts with config:")
    print(f"  Prompts per filler length: {config.n_prompts_per_filler_length}")
    print(f"  Filler lengths (words): {config.filler_word_counts}")
    print(f"  Table entries: {config.n_entries}")
    print(f"  Value digits: {config.value_digits}")
    print(f"  N choices: {config.n_choices}")
    print(f"  Include flipped: {args.include_flipped}")
    base_prompts = len(config.filler_word_counts) * config.n_prompts_per_filler_length
    total_prompts = base_prompts * 2 if args.include_flipped else base_prompts
    print(f"  Total prompts: {total_prompts}")
    print()
    
    prompts = generate_prompt_set(config, include_flipped=args.include_flipped)
    save_prompts(prompts, args.output)
    
    if args.preview:
        print("\n" + "=" * 60)
        print("PREVIEW: First prompt of each filler length")
        print("=" * 60)
        
        seen_lengths = set()
        for p in prompts:
            if p.filler_word_count not in seen_lengths:
                seen_lengths.add(p.filler_word_count)
                print(f"\n--- Filler: {p.filler_word_count} words ---")
                print(f"Query: {p.query_id} → {p.correct_value} (Answer: {p.correct_choice})")
                print(f"Prompt length: {len(p.prompt_text)} chars")
                print(f"Parts: prefix={len(p.prefix)} middle={len(p.middle)} suffix={len(p.suffix)}")
                print()
                # Show structure
                print("PROMPT STRUCTURE:")
                print(f"[PREFIX - {len(p.prefix)} chars]")
                print(p.prefix[:200] + "..." if len(p.prefix) > 200 else p.prefix)
                print(f"[MIDDLE - {len(p.middle)} chars]")
                print(p.middle[:200] + "..." if len(p.middle) > 200 else p.middle)
                print(f"[SUFFIX - {len(p.suffix)} chars]")
                print(p.suffix)


if __name__ == "__main__":
    main()
