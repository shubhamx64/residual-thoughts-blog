"""
Feature Program Composition for Gemma-2 attention heads.

Phase 3 implementation:
- Compose QK routing + OV writing into interpretable programs
- Triplets: (query i) → attend (key j) → write (feature k)
- Program taxonomy: REINFORCE, SHIFT, CROSS-COPY, TRANSFORM, AGGREGATE, BROADCAST, etc.
- Additional patterns: mutual routing, exclusion zones, write collisions
"""

import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from config import Gemma2Config, GEMMA2_CONFIG
from qk_routing import HeadRoutingResult, TopPairs
from ov_writing import HeadWriteResult


class ProgramType(Enum):
    """Classification of feature programs."""
    REINFORCE = "reinforce"        # i → i → i (self-reinforcement)
    SHIFT = "shift"                # i → i → k (attend-self, write-other)
    CROSS_COPY = "cross_copy"      # i → j → j (attend-other, propagate)
    TRANSFORM = "transform"        # i → j → k (full cross-mapping)
    AGGREGATE = "aggregate"        # many i → j → k (fan-in)
    BROADCAST = "broadcast"        # i → j → many k (fan-out)
    SUPPRESS = "suppress"          # i → j → −k (active suppression)
    RELAY = "relay"                # i → j → i (round-trip)
    GATE = "gate"                  # i → i → ±k (conditional write)
    UNKNOWN = "unknown"


@dataclass
class FeatureProgram:
    """A single feature program triplet."""
    query_feature: int      # i: what feature triggers routing
    key_feature: int        # j: what feature gets attended to
    write_feature: int      # k: what feature gets written
    
    route_strength: float   # B[i, j] affinity score
    write_strength: float   # W2F[j, k] write score
    program_score: float    # Combined score
    
    program_type: ProgramType = ProgramType.UNKNOWN
    
    # Whether write is positive or negative
    is_suppressive: bool = False
    
    # Whether this used a fallback self-write (copy_score injection)
    # If True, this shouldn't count as a true REINFORCE
    used_fallback_write: bool = False


@dataclass
class AdditionalPattern:
    """Non-triplet patterns we can mine."""
    pattern_type: str
    features: List[int]
    score: float
    description: str


@dataclass
class HeadProgramResult:
    """Complete program analysis for a single head."""
    layer_idx: int
    query_head: int
    
    # Top programs
    top_programs: List[FeatureProgram]
    
    # Program type distribution
    program_counts: Dict[ProgramType, int]
    
    # Additional patterns (no defaults)
    mutual_routing_pairs: List[Tuple[int, int, float]]     # i↔j
    exclusion_pairs: List[Tuple[int, int, float]]          # strong negative B
    aggregate_targets: List[Tuple[int, List[int], float]]  # (target, sources, strength)
    broadcast_sources: List[Tuple[int, List[int], float]]  # (source, targets, strength)
    
    # Fields with defaults (must come last in dataclass)
    # How many REINFORCE used fallback self-write (not real evidence)
    fallback_reinforce_count: int = 0


def classify_program(
    i: int, j: int, k: int,
    write_strength: float,
) -> ProgramType:
    """
    Classify a triplet (i, j, k) into a program type.
    """
    is_suppressive = write_strength < 0
    
    if is_suppressive:
        return ProgramType.SUPPRESS
    
    if i == j == k:
        return ProgramType.REINFORCE
    
    if i == j and k != i:
        return ProgramType.SHIFT
    
    if i != j and j == k:
        return ProgramType.CROSS_COPY
    
    if i != j and j != k and i == k:
        return ProgramType.RELAY
    
    if i != j and j != k and i != k:
        return ProgramType.TRANSFORM
    
    return ProgramType.UNKNOWN


def compose_programs(
    routing_result: HeadRoutingResult,
    write_result: HeadWriteResult,
    top_k_routes: int = 100,
    top_k_writes_per_key: int = 5,
    route_quantile: float = 0.9,       # Use top 10% of routes
    write_quantile: float = 0.8,       # Use top 20% of writes
    use_quantile_thresholds: bool = True,  # If False, use legacy fixed thresholds
    min_route_score: float = 0.1,      # Only used if use_quantile_thresholds=False
    min_write_score: float = 0.1,      # Only used if use_quantile_thresholds=False
) -> List[FeatureProgram]:
    """
    Compose QK routing pairs with OV writes to form programs.
    
    Uses QUANTILE-BASED thresholds by default so programs don't silently
    become empty when logit scales change.
    
    Args:
        routing_result: QK routing analysis result
        write_result: OV writing analysis result
        top_k_routes: Number of top routing pairs to consider
        top_k_writes_per_key: Number of top writes per key feature
        route_quantile: Quantile threshold for route inclusion (0.9 = top 10%)
        write_quantile: Quantile threshold for write inclusion (0.8 = top 20%)
        use_quantile_thresholds: If True, compute thresholds from data quantiles
        min_route_score: Fixed threshold (only if use_quantile_thresholds=False)
        min_write_score: Fixed threshold (only if use_quantile_thresholds=False)
        
    Returns:
        List of FeaturePrograms sorted by combined score
    """
    programs = []
    
    # Get top routing pairs (i → j)
    routing_pairs = routing_result.top_pairs.positive_pairs[:top_k_routes]
    
    if not routing_pairs:
        return programs
    
    # Compute quantile-based thresholds from the data
    route_scores = [r[2] for r in routing_pairs]
    if use_quantile_thresholds and route_scores:
        route_threshold = sorted(route_scores)[int(len(route_scores) * (1 - route_quantile))]
    else:
        route_threshold = min_route_score
    
    # Get write mappings (j → k)
    write_pairs = write_result.metrics.top_transform_pairs
    
    # Compute write threshold
    write_scores = [abs(w[2]) for w in write_pairs] if write_pairs else []
    if use_quantile_thresholds and write_scores:
        write_threshold = sorted(write_scores)[int(len(write_scores) * (1 - write_quantile))]
    else:
        write_threshold = min_write_score
    
    # Build key → writes lookup with explicit tracking of which are real vs fallback
    key_to_writes: Dict[int, List[Tuple[int, float, bool]]] = defaultdict(list)  # (k, score, is_fallback)
    
    for j, k, score in write_pairs:
        if abs(score) >= write_threshold:
            key_to_writes[j].append((k, score, False))  # Real write
    
    # Track which keys got fallback self-writes (for copy_dominance > threshold)
    fallback_keys = set()
    if write_result.metrics.copy_dominance > 0.05:
        for i, j, r_score in routing_pairs:
            if j not in key_to_writes:
                # This is a FALLBACK self-write using copy_score
                key_to_writes[j].append((j, write_result.metrics.copy_score, True))
                fallback_keys.add(j)
    
    # Compose triplets
    for i, j, route_score in routing_pairs:
        if route_score < route_threshold:
            continue
        
        writes_for_j = key_to_writes.get(j, [])
        
        # If no explicit writes, add fallback self-write
        used_fallback = False
        if not writes_for_j:
            writes_for_j = [(j, write_result.metrics.copy_score, True)]  # Fallback
            used_fallback = True
        
        for k, write_score, is_fallback_write in writes_for_j[:top_k_writes_per_key]:
            program_score = route_score * abs(write_score)
            program_type = classify_program(i, j, k, write_score)
            
            # Mark as fallback if using synthetic self-write
            is_fallback = is_fallback_write or used_fallback
            
            program = FeatureProgram(
                query_feature=i,
                key_feature=j,
                write_feature=k,
                route_strength=route_score,
                write_strength=write_score,
                program_score=program_score,
                program_type=program_type,
                is_suppressive=(write_score < 0),
                used_fallback_write=is_fallback,
            )
            programs.append(program)
    
    # Sort by program score
    programs.sort(key=lambda p: p.program_score, reverse=True)
    
    return programs


def find_mutual_routing(
    routing_result: HeadRoutingResult,
    threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """
    Find pairs where B[i,j] and B[j,i] are both high.
    
    These are bidirectional attention affinity pairs.
    """
    mutual_pairs = []
    
    # Get positive pairs as a dict for quick lookup
    pairs_dict: Dict[Tuple[int, int], float] = {}
    for i, j, score in routing_result.top_pairs.positive_pairs:
        pairs_dict[(i, j)] = score
    
    # Find mutual pairs
    seen = set()
    for (i, j), score_ij in pairs_dict.items():
        if (j, i) in pairs_dict and (j, i) not in seen:
            score_ji = pairs_dict[(j, i)]
            mutual_score = min(score_ij, score_ji)
            if mutual_score > threshold:
                mutual_pairs.append((i, j, mutual_score))
                seen.add((i, j))
                seen.add((j, i))
    
    mutual_pairs.sort(key=lambda x: x[2], reverse=True)
    return mutual_pairs


def find_exclusion_zones(
    routing_result: HeadRoutingResult,
    threshold: float = -0.3,
) -> List[Tuple[int, int, float]]:
    """
    Find feature pairs with strongly negative B.
    
    These are pairs the head actively avoids routing.
    """
    exclusions = []
    
    for i, j, score in routing_result.top_pairs.negative_pairs:
        if score < threshold:
            exclusions.append((i, j, score))
    
    return exclusions


def find_aggregate_patterns(
    programs: List[FeatureProgram],
    min_sources: int = 3,
) -> List[Tuple[int, List[int], float]]:
    """
    Find targets that receive from multiple query features (fan-in).
    
    Returns: List of (target_k, [source_i_1, source_i_2, ...], avg_strength)
    """
    target_sources: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    for prog in programs:
        if prog.program_score > 0:
            # Group by (key, write) pair
            key = (prog.key_feature, prog.write_feature)
            target_sources[key].append((prog.query_feature, prog.program_score))
    
    aggregates = []
    for (key_j, write_k), sources in target_sources.items():
        if len(sources) >= min_sources:
            source_features = [s[0] for s in sources]
            avg_strength = sum(s[1] for s in sources) / len(sources)
            aggregates.append((write_k, source_features, avg_strength))
    
    aggregates.sort(key=lambda x: len(x[1]), reverse=True)
    return aggregates


def find_broadcast_patterns(
    programs: List[FeatureProgram],
    min_targets: int = 3,
) -> List[Tuple[int, List[int], float]]:
    """
    Find sources that write to multiple features (fan-out).
    
    Returns: List of (source_i, [target_k_1, target_k_2, ...], avg_strength)
    """
    source_targets: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    for prog in programs:
        if prog.program_score > 0:
            key = (prog.query_feature, prog.key_feature)
            source_targets[key].append((prog.write_feature, prog.program_score))
    
    broadcasts = []
    for (query_i, key_j), targets in source_targets.items():
        if len(targets) >= min_targets:
            target_features = [t[0] for t in targets]
            avg_strength = sum(t[1] for t in targets) / len(targets)
            broadcasts.append((query_i, target_features, avg_strength))
    
    broadcasts.sort(key=lambda x: len(x[1]), reverse=True)
    return broadcasts


def analyze_head_programs(
    routing_result: HeadRoutingResult,
    write_result: HeadWriteResult,
    top_k_programs: int = 100,
) -> HeadProgramResult:
    """
    Complete program analysis for a single head.
    
    Args:
        routing_result: QK routing analysis
        write_result: OV writing analysis
        top_k_programs: Number of top programs to keep
        
    Returns:
        HeadProgramResult with all patterns
    """
    # Compose programs
    programs = compose_programs(routing_result, write_result)
    top_programs = programs[:top_k_programs]
    
    # Count program types, but exclude fallback writes from REINFORCE
    # Fallback REINFORCE means we had no explicit write evidence - just assumed self-write
    program_counts: Dict[ProgramType, int] = defaultdict(int)
    fallback_reinforce_count = 0
    
    for prog in top_programs:
        # If it's REINFORCE but used fallback write, track separately
        if prog.program_type == ProgramType.REINFORCE and prog.used_fallback_write:
            fallback_reinforce_count += 1
            # Still count in program_counts but we'll report separately
            program_counts[prog.program_type] += 1
        else:
            program_counts[prog.program_type] += 1
    
    # Find additional patterns
    mutual_pairs = find_mutual_routing(routing_result)
    exclusion_pairs = find_exclusion_zones(routing_result)
    aggregate_targets = find_aggregate_patterns(programs)
    broadcast_sources = find_broadcast_patterns(programs)
    
    return HeadProgramResult(
        layer_idx=routing_result.layer_idx,
        query_head=routing_result.query_head,
        top_programs=top_programs,
        program_counts=dict(program_counts),
        fallback_reinforce_count=fallback_reinforce_count,
        mutual_routing_pairs=mutual_pairs,
        exclusion_pairs=exclusion_pairs,
        aggregate_targets=aggregate_targets,
        broadcast_sources=broadcast_sources,
    )


def summarize_programs(result: HeadProgramResult) -> str:
    """Generate a human-readable summary of program analysis."""
    lines = []
    lines.append(f"Layer {result.layer_idx}, Head {result.query_head}")
    lines.append("-" * 40)
    
    # Program type distribution
    lines.append("Program types:")
    for ptype, count in sorted(result.program_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {ptype.value}: {count}")
    
    # Top programs
    lines.append("\nTop 5 programs:")
    for prog in result.top_programs[:5]:
        lines.append(f"  {prog.query_feature} → {prog.key_feature} → {prog.write_feature} "
                    f"({prog.program_type.value}, score={prog.program_score:.3f})")
    
    # Patterns
    if result.mutual_routing_pairs:
        lines.append(f"\nMutual routing pairs: {len(result.mutual_routing_pairs)}")
        for i, j, s in result.mutual_routing_pairs[:3]:
            lines.append(f"  {i} ↔ {j} (score={s:.3f})")
    
    if result.exclusion_pairs:
        lines.append(f"\nExclusion pairs: {len(result.exclusion_pairs)}")
    
    if result.aggregate_targets:
        lines.append(f"\nAggregate targets: {len(result.aggregate_targets)}")
        for k, sources, s in result.aggregate_targets[:2]:
            lines.append(f"  → {k}: {len(sources)} sources")
    
    if result.broadcast_sources:
        lines.append(f"\nBroadcast sources: {len(result.broadcast_sources)}")
        for i, targets, s in result.broadcast_sources[:2]:
            lines.append(f"  {i} →: {len(targets)} targets")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Feature Programs module loaded.")
    print(f"Program types: {[p.value for p in ProgramType]}")
