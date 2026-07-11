import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from config import RNG_SEED

# -----------------------------
# Data structures
# -----------------------------


@dataclass
class LayerRegion:
    """Inclusive layer range in hidden_state indexing (1..N, where 0 is embeddings)."""

    start: int
    end: int

    def to_module_indices(self) -> List[int]:
        """
        Convert to module indices (0-based transformer block list).
        Example: hidden_state layer 1 -> module index 0.
        """
        return list(range(max(0, self.start - 1), max(0, self.end)))


@dataclass
class SteeringProfile:
    name: str
    direction_id: str
    alpha: float
    region: LayerRegion
    scope: str  # "full" | "completion"
    operator_aware: bool = False
    operator_override: Optional[str] = None  # add | sub | mul | div | None


@dataclass
class SteeringConfig:
    profile_name: str
    alpha: float
    region_preset: str
    scope: str
    custom_range: Optional[Sequence[int]] = None
    operator_aware: bool = False
    include_baseline: bool = False


# -----------------------------
# Presets and utilities
# -----------------------------


_DIRECTION_CACHE: Dict[str, torch.Tensor] = {}
_CUSTOM_METADATA: Dict[str, dict] = {}
_PROFILE_PATH = Path(os.environ.get("STEERING_PROFILE_PATH", Path(__file__).resolve().parent / "steering_profiles.json"))

# Keyword seed lists for direction construction
_SEED_KEYWORDS = {
    "perspective_cs_vs_lit_pos": ["computer", "algorithm", "data", "network", "binary"],
    "perspective_cs_vs_lit_neg": ["poem", "metaphor", "character", "plot", "novel"],
    "syntax_short_vs_long_pos": ["concise", "brief", "succinct", "summary"],
    "syntax_short_vs_long_neg": ["elaborate", "detailed", "comprehensive", "extended"],
    "math_reason_pos": ["therefore", "compute", "step", "calculate", "derive"],
    "math_reason_neg": ["maybe", "guess", "perhaps", "probably"],
    "math_add_pos": ["add", "plus", "sum", "together"],
    "math_sub_pos": ["subtract", "minus", "difference", "left"],
    "math_mul_pos": ["multiply", "times", "product", "groups"],
}

_DEFAULT_PROFILE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "perspective",
        "label": "Perspective (CS vs Lit)",
        "pos_prompts": _SEED_KEYWORDS["perspective_cs_vs_lit_pos"],
        "neg_prompts": _SEED_KEYWORDS["perspective_cs_vs_lit_neg"],
        "type": "builtin",
    },
    {
        "id": "syntax",
        "label": "Syntax (short vs long)",
        "pos_prompts": _SEED_KEYWORDS["syntax_short_vs_long_pos"],
        "neg_prompts": _SEED_KEYWORDS["syntax_short_vs_long_neg"],
        "type": "builtin",
    },
    {
        "id": "math",
        "label": "Math (reasoned CoT)",
        "pos_prompts": _SEED_KEYWORDS["math_reason_pos"],
        "neg_prompts": _SEED_KEYWORDS["math_reason_neg"],
        "type": "builtin",
    },
    {"id": "math_add", "label": "Math (add)", "pos_prompts": _SEED_KEYWORDS["math_add_pos"], "neg_prompts": [], "type": "builtin"},
    {"id": "math_sub", "label": "Math (sub)", "pos_prompts": _SEED_KEYWORDS["math_sub_pos"], "neg_prompts": [], "type": "builtin"},
    {"id": "math_mul", "label": "Math (mul)", "pos_prompts": _SEED_KEYWORDS["math_mul_pos"], "neg_prompts": [], "type": "builtin"},
]


def _get_layers_list(model) -> List[torch.nn.Module]:
    """
    Try to fetch the list of transformer blocks for Gemma/Llama-like decoders.
    Falls back to common attribute names.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Unable to locate transformer layers for steering hooks.")


def get_hook_indices_for_profile(profile: SteeringProfile, model) -> List[int]:
    layers = _get_layers_list(model)
    total_layers = len(layers)
    return [i for i in profile.region.to_module_indices() if 0 <= i < total_layers]


def _region_from_preset(total_layers: int, preset: str, custom_range: Optional[Sequence[int]]) -> LayerRegion:
    """
    Map preset -> inclusive hidden_state layers.
    Hidden_state indexing: embeddings=0, block1=1, ..., blockN=N.
    """
    preset = (preset or "corridor").lower()
    if preset == "custom" and custom_range:
        start, end = int(custom_range[0]), int(custom_range[-1])
        start = max(1, min(total_layers, start))
        end = max(start, min(total_layers, end))
        return LayerRegion(start=start, end=end)

    # heuristic splits
    early_end = max(1, math.ceil(total_layers * 0.25))
    late_start = max(1, math.floor(total_layers * 0.75))
    corridor_start = max(1, math.floor(total_layers * 0.35))
    corridor_end = max(corridor_start, math.ceil(total_layers * 0.65))

    if preset == "early":
        return LayerRegion(1, early_end)
    if preset == "late":
        return LayerRegion(late_start, total_layers)
    # default corridor
    return LayerRegion(corridor_start, corridor_end)


def _build_direction_from_keywords(
    direction_id: str,
    tokenizer,
    model,
    pos_keywords: List[str],
    neg_keywords: List[str],
) -> torch.Tensor:
    """
    Build a deterministic direction from token embedding means of keyword sets.
    If a keyword is unknown, it is skipped.
    """
    if direction_id in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[direction_id]

    emb = model.get_input_embeddings().weight  # [vocab, hidden]
    device = emb.device
    dtype = emb.dtype

    def embed_keywords(words: List[str]) -> torch.Tensor:
        ids = []
        for w in words:
            toks = tokenizer.encode(w, add_special_tokens=False)
            ids.extend(toks)
        if not ids:
            return torch.zeros(emb.shape[1], device=device, dtype=dtype)
        vecs = emb[torch.tensor(ids, device=device)]
        return vecs.mean(dim=0)

    pos_vec = embed_keywords(pos_keywords)
    neg_vec = embed_keywords(neg_keywords)
    direction = pos_vec - neg_vec
    norm = torch.norm(direction) + 1e-8
    direction = direction / norm
    _DIRECTION_CACHE[direction_id] = direction
    return direction


def _seeded_random_direction(direction_id: str, hidden_size: int, device, dtype) -> torch.Tensor:
    """
    Deterministic fallback if keyword-derived vector is degenerate.
    """
    g = torch.Generator(device=device)
    g.manual_seed(abs(hash((direction_id, RNG_SEED))) % (2**31))
    vec = torch.randn(hidden_size, generator=g, device=device, dtype=dtype)
    return vec / (torch.norm(vec) + 1e-8)


def _build_direction_from_prompts(
    direction_id: str,
    tokenizer,
    model,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> torch.Tensor:
    """
    Build a direction from arbitrary positive vs negative prompts.
    Tokenizes full prompts and averages input embeddings.
    """
    if direction_id in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[direction_id]

    emb = model.get_input_embeddings().weight  # [vocab, hidden]
    device = emb.device
    dtype = emb.dtype

    def embed_texts(texts: List[str]) -> torch.Tensor:
        ids = []
        for t in texts:
            toks = tokenizer.encode(t, add_special_tokens=False)
            ids.extend(toks)
        if not ids:
            return torch.zeros(emb.shape[1], device=device, dtype=dtype)
        vecs = emb[torch.tensor(ids, device=device)]
        return vecs.mean(dim=0)

    pos_vec = embed_texts(pos_prompts)
    neg_vec = embed_texts(neg_prompts)
    direction = pos_vec - neg_vec
    norm = torch.norm(direction) + 1e-8
    direction = direction / norm
    _DIRECTION_CACHE[direction_id] = direction
    return direction


def _capture_hidden_mean(
    prompt: str,
    completion: str,
    layers: List[int],
    scope: str,
    tokenizer,
    model,
) -> torch.Tensor:
    """
    Run the model on prompt+completion and average hidden states over selected layers/tokens.
    scope: "completion" (tokens after prompt) or "full".
    """
    text = prompt + ("\n" + completion if completion else "")
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    prompt_len = enc_prompt["input_ids"].shape[1]
    enc = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = out.hidden_states  # tuple len L_total
    seq_len = enc["input_ids"].shape[1]
    token_mask = torch.ones(seq_len, device=enc["input_ids"].device, dtype=torch.bool)
    if scope == "completion":
        token_mask[:prompt_len] = False

    vecs = []
    for li in layers:
        if li < 0 or li >= len(hs):
            continue
        h = hs[li][0]  # [seq, hidden]
        h_sel = h[token_mask]
        if h_sel.numel() == 0:
            continue
        vecs.append(h_sel.mean(dim=0, keepdim=True))
    if not vecs:
        return torch.zeros(model.config.hidden_size, device=enc["input_ids"].device, dtype=next(model.parameters()).dtype)
    stacked = torch.cat(vecs, dim=0).mean(dim=0)  # [hidden]
    return stacked


def _slugify(name: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "custom"


def register_custom_direction(
    name: str,
    pos_prompts: List[str],
    neg_prompts: List[str],
    tokenizer,
    model,
) -> Dict[str, Any]:
    """
    Register a custom steering direction built from user prompts.
    Returns metadata for UI listing.
    """
    slug = _slugify(name)
    direction_id = f"custom:{slug}"
    vec = _build_direction_from_prompts(direction_id, tokenizer, model, pos_prompts, neg_prompts)
    # cache already set; store metadata
    meta = {
        "id": direction_id,
        "label": name,
        "type": "custom",
        "mode": "embedding",
        "pos_prompts": pos_prompts,
        "neg_prompts": neg_prompts,
    }
    _CUSTOM_METADATA[direction_id] = meta
    _DIRECTION_CACHE[direction_id] = vec
    _persist_custom_profiles()
    return meta


def register_teacher_forced_direction(
    name: str,
    pos_examples: List[Tuple[str, str]],
    neg_examples: List[Tuple[str, str]],
    layers: List[int],
    scope: str,
    tokenizer,
    model,
) -> Dict[str, Any]:
    """
    Build direction from hidden states on completed examples (teacher forcing).
    pos/neg_examples: list of (prompt, completion) strings.
    """
    slug = _slugify(name)
    direction_id = f"custom:{slug}"
    scope_use = "completion" if scope == "completion" else "full"
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def mean_vec(examples: List[Tuple[str, str]]) -> torch.Tensor:
        acc = []
        for prompt, comp in examples:
            v = _capture_hidden_mean(prompt, comp, layers, scope_use, tokenizer, model)
            acc.append(v.to(torch.float32))
        if not acc:
            return torch.zeros(model.config.hidden_size, device=device, dtype=dtype)
        m = torch.stack(acc, dim=0).mean(dim=0)
        return m

    pos_vec = mean_vec(pos_examples)
    neg_vec = mean_vec(neg_examples) if neg_examples else torch.zeros_like(pos_vec)
    direction = pos_vec - neg_vec
    norm = torch.norm(direction) + 1e-8
    direction = (direction / norm).to(device=device, dtype=dtype)

    meta = {
        "id": direction_id,
        "label": name,
        "type": "custom_teacher",
        "mode": "teacher_forced",
        "pos_examples": pos_examples,
        "neg_examples": neg_examples,
        "layers": layers,
        "scope": scope_use,
    }
    _CUSTOM_METADATA[direction_id] = meta
    _DIRECTION_CACHE[direction_id] = direction
    _persist_custom_profiles()
    return meta


def get_direction_vector(direction_id: str, tokenizer, model) -> torch.Tensor:
    """
    Retrieve or build a steering direction. Uses keyword deltas with a random fallback.
    """
    if direction_id in _DIRECTION_CACHE:
        vec = _DIRECTION_CACHE[direction_id]
        # move to correct device/dtype if needed
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return vec.to(device=device, dtype=dtype)

    hidden_size = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if direction_id in _CUSTOM_METADATA:
        meta = _CUSTOM_METADATA[direction_id]
        if meta.get("mode") == "teacher_forced":
            layers = meta.get("layers") or []
            scope = meta.get("scope", "completion")
            pos = meta.get("pos_examples", [])
            neg = meta.get("neg_examples", [])
            pos_pairs = [(p, c) for p, c in pos] if pos and isinstance(pos[0], (list, tuple)) else [(d.get("prompt", ""), d.get("completion", "")) for d in pos]
            neg_pairs = [(p, c) for p, c in neg] if neg and isinstance(neg[0], (list, tuple)) else [(d.get("prompt", ""), d.get("completion", "")) for d in neg]
            vec = register_teacher_forced_direction(
                name=meta.get("label", direction_id),
                pos_examples=pos_pairs,
                neg_examples=neg_pairs,
                layers=layers,
                scope=scope,
                tokenizer=tokenizer,
                model=model,
            )
            return _DIRECTION_CACHE[direction_id]
        else:
            vec = _build_direction_from_prompts(
                direction_id,
                tokenizer,
                model,
                meta.get("pos_prompts", []),
                meta.get("neg_prompts", []),
            )
            return vec

    if direction_id == "perspective_cs_vs_lit":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["perspective_cs_vs_lit_pos"],
            _SEED_KEYWORDS["perspective_cs_vs_lit_neg"],
        )
    elif direction_id == "syntax_short_vs_long":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["syntax_short_vs_long_pos"],
            _SEED_KEYWORDS["syntax_short_vs_long_neg"],
        )
    elif direction_id == "math_reason":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["math_reason_pos"],
            _SEED_KEYWORDS["math_reason_neg"],
        )
    elif direction_id == "math_add":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["math_add_pos"],
            [],
        )
    elif direction_id == "math_sub":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["math_sub_pos"],
            [],
        )
    elif direction_id == "math_mul":
        vec = _build_direction_from_keywords(
            direction_id,
            tokenizer,
            model,
            _SEED_KEYWORDS["math_mul_pos"],
            [],
        )
    else:
        vec = torch.zeros(hidden_size, device=device, dtype=dtype)

    if vec.abs().sum() < 1e-6:
        vec = _seeded_random_direction(direction_id, hidden_size, device, dtype)

    vec = vec.to(device=device, dtype=dtype)
    _DIRECTION_CACHE[direction_id] = vec
    return vec


def build_profile(
    name: str,
    alpha: float,
    region_preset: str,
    scope: str,
    total_layers: int,
    custom_range: Optional[Sequence[int]] = None,
    operator_aware: bool = False,
    operator_override: Optional[str] = None,
) -> Optional[SteeringProfile]:
    """
    Factory for steering profiles. Returns None for "none".
    """
    lname = (name or "none").lower()
    if lname == "none":
        return None

    direction_id = "perspective_cs_vs_lit"
    if lname.startswith("perspective"):
        direction_id = "perspective_cs_vs_lit"
    elif lname.startswith("syntax"):
        direction_id = "syntax_short_vs_long"
    elif lname.startswith("math_add"):
        direction_id = "math_add"
    elif lname.startswith("math_sub"):
        direction_id = "math_sub"
    elif lname.startswith("math_mul"):
        direction_id = "math_mul"
    elif lname.startswith("math"):
        direction_id = "math_reason"
    elif lname.startswith("custom"):
        direction_id = lname

    # Override if operator-aware wants a specific math op
    if operator_override in ("add", "sub", "mul"):
        direction_id = f"math_{operator_override}"

    region = _region_from_preset(total_layers, region_preset, custom_range)
    resolved_scope = "completion" if scope == "completion" else "full"
    return SteeringProfile(
        name=lname,
        direction_id=direction_id,
        alpha=float(alpha),
        region=region,
        scope=resolved_scope,
        operator_aware=operator_aware,
        operator_override=operator_override,
    )


# -----------------------------
# Operator detection & metrics
# -----------------------------


def list_available_profiles() -> List[Dict[str, Any]]:
    """
    Return builtin profiles plus user-defined custom ones with their seed prompts.
    """
    return _DEFAULT_PROFILE_CATALOG + list(_CUSTOM_METADATA.values())


def _persist_custom_profiles():
    try:
        data = list(_CUSTOM_METADATA.values())
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[steering] Warning: failed to persist profiles to {_PROFILE_PATH}: {e}")


def _load_custom_profiles():
    if not _PROFILE_PATH.exists():
        return
    try:
        raw = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for meta in raw:
                if not isinstance(meta, dict):
                    continue
                pid = meta.get("id")
                if not pid:
                    continue
                _CUSTOM_METADATA[pid] = meta
    except Exception as e:
        print(f"[steering] Warning: failed to load profiles from {_PROFILE_PATH}: {e}")


def parse_layer_range(layer_str: str, total_layers: int) -> List[int]:
    """
    Parse a string like '8-13' or '3,5,7' into a list of layer indices.
    """
    if not layer_str:
        return []
    parts = []
    for chunk in layer_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                a, b = chunk.split("-", 1)
                a = int(a)
                b = int(b)
                if a > b:
                    a, b = b, a
                parts.extend(list(range(a, b + 1)))
            except Exception:
                continue
        else:
            try:
                parts.append(int(chunk))
            except Exception:
                continue
    # clamp to valid range
    parts = [p for p in parts if 0 <= p <= total_layers]
    # unique and sorted
    return sorted(list(dict.fromkeys(parts)))


# Load persisted custom profiles at import time
_load_custom_profiles()


def detect_operator_tag(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in [" add", " plus", "sum", "+", "more than"]):
        return "add"
    if any(k in t for k in [" subtract", " minus", "difference", "- "]):
        return "sub"
    if any(k in t for k in ["multiply", "times", "product", "groups", " x "]):
        return "mul"
    if any(k in t for k in [" divide", "quotient", "ratio", " per ", "/ "]):
        return "div"
    return None


def perspective_score(text: str) -> float:
    """
    Simple CS-vs-Lit keyword differential: positive = CS tilt, negative = Lit tilt.
    """
    cs = ["algorithm", "compute", "data", "binary", "graph", "program", "complexity"]
    lit = ["poem", "novel", "metaphor", "character", "plot", "imagery", "rhyme"]
    txt = (text or "").lower()
    cs_count = sum(txt.count(w) for w in cs)
    lit_count = sum(txt.count(w) for w in lit)
    return float(cs_count - lit_count)


def sentence_length_stats(text: str) -> Dict[str, float]:
    import re

    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sentences:
        return {"avg_len": 0.0, "num_sentences": 0}
    lengths = [len(s.split()) for s in sentences]
    return {
        "avg_len": float(sum(lengths) / len(lengths)),
        "num_sentences": float(len(sentences)),
    }


def reasoning_stepiness(text: str) -> int:
    """
    Count of step-like markers to approximate deliberate reasoning.
    """
    markers = ["step", "first", "second", "third", "then", "next", "finally"]
    txt = (text or "").lower()
    return sum(txt.count(m) for m in markers)


# -----------------------------
# Hooking
# -----------------------------


def register_steering_hooks(
    model,
    tokenizer,
    profile: SteeringProfile,
    prompt_lens: torch.Tensor,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on the chosen layer region.
    Returns handles to remove after generation.
    """
    layers = _get_layers_list(model)
    module_indices = get_hook_indices_for_profile(profile, model)
    if not module_indices:
        return []

    direction = get_direction_vector(profile.direction_id, tokenizer, model)
    direction = direction.to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    dir_scaled = profile.alpha * direction  # [hidden]

    handles: List[torch.utils.hooks.RemovableHandle] = []

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not isinstance(hidden, torch.Tensor):
            return output
        # hidden: [batch, seq, hidden]
        if hidden.dim() != 3:
            return output

        batch, seq, hid = hidden.shape
        if dir_scaled.shape[0] != hid:
            return output

        mask = torch.ones(batch, seq, device=hidden.device, dtype=hidden.dtype)
        if profile.scope == "completion":
            # prompt_lens: [batch]
            seq_range = torch.arange(seq, device=hidden.device).unsqueeze(0)  # [1, seq]
            mask = (seq_range >= prompt_lens.unsqueeze(1)).to(hidden.dtype)

        hidden = hidden + mask.unsqueeze(-1) * dir_scaled.view(1, 1, -1)
        if rest is None:
            return hidden
        return (hidden,) + rest

    for idx in module_indices:
        handles.append(layers[idx].register_forward_hook(hook_fn))
    return handles
