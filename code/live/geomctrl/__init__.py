from .config import ExperimentConfig, parse_args
from .prompts import build_prompt_families, build_cs_lit_prompts
from .geometry import (
    DepthGeometryAggregator,
    compute_corridor_indices,
    collect_geometry_for_family,
    set_seed,
    get_model_tag,
    load_model_and_tokenizer,
)
from .steering import (
    get_decoder_layers,
    build_direction_cs_lit,
    cs_lit_score,
    generate_with_steering,
    run_steering_layer_sweep,
)
from .plots import (
    plot_geometry_condition,
    plot_ci_vs_gain,
)

__all__ = [
    "ExperimentConfig",
    "parse_args",
    "build_prompt_families",
    "build_cs_lit_prompts",
    "DepthGeometryAggregator",
    "compute_corridor_indices",
    "collect_geometry_for_family",
    "set_seed",
    "get_model_tag",
    "load_model_and_tokenizer",
    "get_decoder_layers",
    "build_direction_cs_lit",
    "cs_lit_score",
    "generate_with_steering",
    "run_steering_layer_sweep",
    "plot_geometry_condition",
    "plot_ci_vs_gain",
]

