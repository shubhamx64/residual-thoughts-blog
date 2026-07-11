# model_backend.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, List, Dict, Any
from config import MODEL_ID, DEVICE, DTYPE, RNG_SEED, MAX_NEW_TOKENS
from steering import (
    SteeringProfile,
    build_profile,
    register_steering_hooks,
    sentence_length_stats,
    perspective_score,
    reasoning_stepiness,
)

_tokenizer = None
_model = None


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from typing import List

@torch.inference_mode()
def generate_batch_text(
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> List[str]:
    """
    Batched generation for multiple prompts.
    Uses padding + attention_mask and slices completions based on per-sample prompt length.
    """
    tok, model = get_tokenizer_and_model()
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # number of non-pad tokens per sample
    prompt_lens = attention_mask.sum(dim=1)  # [batch]

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=False,
    )  # [batch, seq_len_out]

    outputs: List[str] = []
    for i in range(gen_ids.size(0)):
        start = int(prompt_lens[i].item())
        completion_ids = gen_ids[i, start:]
        txt = tok.decode(completion_ids, skip_special_tokens=True).strip()
        outputs.append(txt)
    return outputs


@torch.inference_mode()
def generate_batch_text_with_steering(
    prompts: List[str],
    profile: Optional[SteeringProfile],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> List[str]:
    """
    Steering-aware generation. Falls back to vanilla generation if profile is None.
    """
    if profile is None:
        return generate_batch_text(prompts, max_new_tokens=max_new_tokens)

    tok, model = get_tokenizer_and_model()
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_lens = attention_mask.sum(dim=1)  # [batch]

    hooks = register_steering_hooks(
        model=model,
        tokenizer=tok,
        profile=profile,
        prompt_lens=prompt_lens,
    )

    try:
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()

    outputs: List[str] = []
    for i in range(gen_ids.size(0)):
        start = int(prompt_lens[i].item())
        completion_ids = gen_ids[i, start:]
        txt = tok.decode(completion_ids, skip_special_tokens=True).strip()
        outputs.append(txt)
    return outputs


def build_metrics_for_text(text: str) -> Dict[str, Any]:
    """
    Lightweight qualitative metrics used in both playground and GSM8K summaries.
    """
    sent_stats = sentence_length_stats(text)
    return {
        "perspective_score": perspective_score(text),
        "avg_sentence_length": sent_stats.get("avg_len", 0.0),
        "num_sentences": sent_stats.get("num_sentences", 0.0),
        "reasoning_stepiness": reasoning_stepiness(text),
    }

def get_tokenizer_and_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _seed_everything(RNG_SEED)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    _model.eval()

    # Log once
    first_param = next(_model.parameters())
    print(f"[model_backend] Loaded {MODEL_ID} on device={first_param.device}, dtype={first_param.dtype}")

    if not torch.cuda.is_available():
        print("[model_backend] WARNING: torch.cuda.is_available() is False, running on CPU. "
              "Install CUDA-enabled PyTorch to use your 4060 Ti.")

    return _tokenizer, _model

@torch.inference_mode()
def run_model_get_layer_hidden(
    text: str,
    layer_index: int,
) -> Tuple[torch.Tensor, int]:
    """
    Run model on `text` and return:
      - hidden states at specified layer index [seq, hidden]
      - prompt length (number of input tokens, not including generated tokens)
    Uses output_hidden_states=True, use_cache=False for reproducibility.
    """
    tok, model = get_tokenizer_and_model()
    enc = tok(text, return_tensors="pt").to(DEVICE)
    prompt_len = enc["input_ids"].shape[1]

    out = model(
        **enc,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states: tuple length L_total; each [batch, seq, hidden]
    hs = out.hidden_states[layer_index][0]  # [seq, hidden]
    return hs.to(torch.float32), prompt_len


@torch.inference_mode()
def generate_text(text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    tok, model = get_tokenizer_and_model()
    enc = tok(text, return_tensors="pt").to(DEVICE)
    prompt_len = enc["input_ids"].shape[1]

    gen_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=False,
    )[0]
    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion.strip()
