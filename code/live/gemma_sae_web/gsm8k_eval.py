# gsm8k_eval.py
import re
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Any, Tuple

from datasets import load_dataset
from config import GSM8K_DATASET, GSM8K_SPLIT, GSM8K_MAX_EXAMPLES, MAX_NEW_TOKENS, GSM8K_BATCH_SIZE
from model_backend import generate_batch_text, generate_batch_text_with_steering, build_metrics_for_text, get_tokenizer_and_model
from steering import build_profile, detect_operator_tag, get_hook_indices_for_profile

_number_pattern = re.compile(r"-?\d[\d,]*\.?\d*")
_PROGRESS: Dict[str, Any] = {"active": False, "done": 0, "total": 0}
DEFAULT_BATCH_SIZE = 8  # safe default


def extract_last_number(text: str):
    matches = _number_pattern.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def normalize_answer(ans_str: str):
    ans_str = ans_str.strip()
    return extract_last_number(ans_str)


def check_gsm8k_answer(model_output: str, gold_answer: str, tol=Decimal("1e-6")) -> Tuple[bool, Any, Any]:
    model_num = extract_last_number(model_output)
    gold_num = normalize_answer(gold_answer)

    if model_num is None or gold_num is None:
        return False, model_num, gold_num

    diff = abs(model_num - gold_num)
    return diff <= tol, model_num, gold_num


def load_gsm8k_subset(n_examples: int):
    """
    Load up to n_examples from the configured dataset.
    If the requested split does not exist, fall back to the first available split.
    """
    n = min(n_examples, GSM8K_MAX_EXAMPLES)

    # First load builder to see available splits
    builder = load_dataset(GSM8K_DATASET)
    available_splits = list(builder.keys())

    if GSM8K_SPLIT in available_splits:
        split_to_use = GSM8K_SPLIT
    else:
        # Fall back to the first split (for thesven/gsm8k-reasoning this is 'train')
        split_to_use = available_splits[0]

    ds = builder[split_to_use]

    # Just take the first n examples for reproducibility
    ds = ds.select(range(min(n, len(ds))))
    return list(ds)


def evaluate_gsm8k(
    n_examples: int = 50,
    batch_size: int = DEFAULT_BATCH_SIZE,
    steering_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data = load_gsm8k_subset(n_examples)
    total = len(data)
    if total == 0:
        _PROGRESS.update({"active": False, "done": 0, "total": 0})
        return {
            "num_examples": 0,
            "num_correct": 0,
            "accuracy": 0.0,
            "samples": [],
            "records": [],
        }
    _PROGRESS.update({"active": True, "done": 0, "total": total})

    # Prepare prompts
    prompts = []
    gold_answers = []
    questions = []
    for row in data:
        q = row.get("question", row.get("prompt", ""))
        gold = row["answer"]
        questions.append(q)
        gold_answers.append(gold)
        prompts.append(q.strip() + "\n\nLet's think step by step.")

    # Resolve steering profile once (operator-aware is handled per-question later)
    tok, model = get_tokenizer_and_model()
    total_layers = getattr(model.config, "num_hidden_layers", 0)
    steering_profile = None
    operator_aware = False
    include_baseline = False
    region = "corridor"
    scope = "completion"
    alpha = 0.0
    profile_name = "none"
    custom_range = None
    hook_indices: List[int] = []
    steering_applied_flag = False
    if steering_config:
        profile_name = steering_config.get("profile_name", "none")
        alpha = float(steering_config.get("alpha", 0.0))
        region = steering_config.get("region", "corridor")
        scope = steering_config.get("scope", "completion")
        custom_range = steering_config.get("custom_range") or None
        include_baseline = bool(steering_config.get("include_baseline", False))
        operator_aware = bool(steering_config.get("operator_aware", False))
        steering_profile = build_profile(
            profile_name,
            alpha=alpha,
            region_preset=region,
            scope=scope,
            total_layers=total_layers,
            custom_range=custom_range,
            operator_aware=operator_aware,
        )
        if steering_profile is not None:
            hook_indices = get_hook_indices_for_profile(steering_profile, model)
            steering_applied_flag = bool(hook_indices) and alpha != 0.0

    correct_count = 0
    all_results = []
    preview_results = []

    # Operator stats
    op_stats: Dict[str, Dict[str, int]] = {
        "add": {"total": 0, "correct": 0},
        "sub": {"total": 0, "correct": 0},
        "mul": {"total": 0, "correct": 0},
        "div": {"total": 0, "correct": 0},
        "other": {"total": 0, "correct": 0},
    }

    # Batched loop
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_prompts = prompts[start:end]
        batch_gold = gold_answers[start:end]
        batch_questions = questions[start:end]

        batch_profiles = []
        if operator_aware and steering_profile is not None:
            # choose direction based on operator tag
            for q in batch_questions:
                op = detect_operator_tag(q)
                override = None
                if op in ("add", "sub", "mul"):
                    override = f"math_{op}"
                p = build_profile(
                    profile_name if override is None else override,
                    alpha=alpha,
                    region_preset=region,
                    scope=scope,
                    total_layers=total_layers,
                    custom_range=custom_range,
                    operator_aware=operator_aware,
                    operator_override=op,
                )
                batch_profiles.append(p)
        else:
            batch_profiles = [steering_profile] * len(batch_prompts)

        completions = []
        if steering_profile is None:
            completions = generate_batch_text(batch_prompts, max_new_tokens=MAX_NEW_TOKENS)
        else:
            # run one by one if operator-aware; otherwise in batch
            if operator_aware:
                for p_prompt, p_profile in zip(batch_prompts, batch_profiles):
                    comp = generate_batch_text_with_steering([p_prompt], profile=p_profile, max_new_tokens=MAX_NEW_TOKENS)[0]
                    completions.append(comp)
            else:
                completions = generate_batch_text_with_steering(batch_prompts, profile=steering_profile, max_new_tokens=MAX_NEW_TOKENS)

        baseline_outputs = []
        if include_baseline and steering_profile is not None:
            baseline_outputs = generate_batch_text(batch_prompts, max_new_tokens=MAX_NEW_TOKENS)

        for i, completion in enumerate(completions):
            idx = start + i
            gold = batch_gold[i]
            q = batch_questions[i]
            op_tag = detect_operator_tag(q) or "other"

            is_correct, model_num, gold_num = check_gsm8k_answer(completion, gold)
            if is_correct:
                correct_count += 1
            op_stats.setdefault(op_tag, {"total": 0, "correct": 0})
            op_stats[op_tag]["total"] += 1
            if is_correct:
                op_stats[op_tag]["correct"] += 1

            baseline_output = baseline_outputs[i] if baseline_outputs else None
            baseline_correct = None
            if baseline_output is not None:
                baseline_correct, _, _ = check_gsm8k_answer(baseline_output, gold)

            flip_type = None
            if baseline_correct is not None:
                if baseline_correct and not is_correct:
                    flip_type = "correct_to_incorrect"
                elif (not baseline_correct) and is_correct:
                    flip_type = "incorrect_to_correct"
                else:
                    flip_type = "unchanged"

            record = {
                "index": idx,
                "question": q,
                "gold_answer_raw": gold,
                "gold_answer_parsed": str(gold_num) if gold_num is not None else None,
                "model_output": completion,
                "model_answer_parsed": str(model_num) if model_num is not None else None,
                "is_correct": bool(is_correct),
                "baseline_output": baseline_output,
                "baseline_correct": baseline_correct,
                "flip_type": flip_type,
                "steering_profile": profile_name,
                "steering_alpha": alpha,
                "steering_region": region,
                "steering_scope": scope,
                "steering_applied": steering_applied_flag,
                "operator_tag": op_tag,
            }
            all_results.append(record)

            if len(preview_results) < 10:
                preview_results.append(record)
        _PROGRESS.update({"active": True, "done": end, "total": total})

    accuracy = correct_count / total
    steering_meta = {
        "profile": profile_name,
        "alpha": alpha,
        "region": region,
        "scope": scope,
        "operator_aware": operator_aware,
        "include_baseline": include_baseline,
        "hook_indices": hook_indices,
        "num_hook_layers": len(hook_indices),
        "steering_applied": steering_applied_flag,
    }

    breakdown = []
    for op, stat in op_stats.items():
        total_op = max(1, stat["total"])
        breakdown.append(
            {
                "operator": op,
                "total": stat["total"],
                "correct": stat["correct"],
                "accuracy": float(stat["correct"] / total_op),
            }
        )

    return {
        "num_examples": int(total),
        "num_correct": int(correct_count),
        "accuracy": float(accuracy),
        "samples": preview_results,
        "records": all_results,
        "steering": steering_meta,
        "operator_breakdown": breakdown,
    }


def get_gsm8k_progress() -> Dict[str, Any]:
    return dict(_PROGRESS)
