#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_MAP = {
    "1B": "google/gemma-3-1b-it",
    "4B": "google/gemma-3-4b-it",
}

DEFAULT_SYSTEM_MESSAGE = "You are Gemma-3, a large language model trained by Google."


def sanitize_for_filename(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def build_run_filename(
    model_choice: str,
    params: Dict[str, Any],
    prefix: str = "completions",
    ext: str = ".csv",
) -> str:
    keys = [
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "do_sample",
        "repetition_penalty",
        "seed",
    ]
    parts = [prefix, f"model={model_choice}"]
    for k in keys:
        if k in params:
            parts.append(f"{k}={params[k]}")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return sanitize_for_filename("__".join(parts)) + ext


def pick_prompt_column(fieldnames: List[str]) -> str:
    for c in ["user_content", "prompt", "text", "input", "instruction"]:
        if c in fieldnames:
            return c
    raise ValueError(
        f"Could not find a prompt column. Expected one of "
        f"['user_content','prompt','text','input','instruction'] but got: {fieldnames}"
    )


def load_model(model_choice: str, torch_dtype: torch.dtype, device_map: str):
    if model_choice not in MODEL_MAP:
        raise ValueError(f"Invalid model choice: {model_choice}")

    model_id = MODEL_MAP[model_choice]
    print(f"[load] Loading {model_choice} ({model_id}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()
    print("[load] Done.")
    return tokenizer, model


@torch.no_grad()
def generate_one(tokenizer, model, system_message: str, user_message: str, gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    chat_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = model.device
    inputs = inputs.to(device)
    attention_mask = torch.ones_like(inputs, device=device)

    input_len = inputs.shape[-1]
    t0 = time.time()

    output_ids = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )

    dt = time.time() - t0
    full = output_ids[0]
    completion_ids = full[input_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    prompt_tokens = int(input_len)
    completion_tokens = int(completion_ids.shape[-1])
    total_tokens = prompt_tokens + completion_tokens

    return {
        "completion_text": completion_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "generation_time_s": dt,
    }


def run_for_one_model(args, model_choice: str):
    # Output goes into <output_dir>/<model_choice>/
    model_out_dir = os.path.join(args.output_dir, model_choice)
    os.makedirs(model_out_dir, exist_ok=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    tokenizer, model = load_model(model_choice, torch_dtype=torch_dtype, device_map=args.device_map)

    do_sample = bool(args.do_sample)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "top_p": args.top_p if do_sample else None,
        "top_k": args.top_k if do_sample and args.top_k > 0 else None,
        "repetition_penalty": args.repetition_penalty,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    run_params_for_name = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }

    out_name = build_run_filename(model_choice, run_params_for_name, prefix="completions")
    out_csv = os.path.join(model_out_dir, out_name)
    out_json = out_csv.replace(".csv", ".runmeta.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "model_choice": model_choice,
                "model_id": MODEL_MAP[model_choice],
                "input_csv": os.path.abspath(args.input_csv),
                "system_message": args.system_message,
                "gen_kwargs": gen_kwargs,
                "seed": args.seed,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "notes": "Output CSV contains original prompt columns + completion metadata columns.",
            },
            f,
            indent=2,
        )

    with open(args.input_csv, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        in_fieldnames = reader.fieldnames or []
        prompt_col = pick_prompt_column(in_fieldnames)

        extra_cols = [
            "completion_text",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "generation_time_s",
            "error",
        ]
        out_fieldnames = list(in_fieldnames) + [c for c in extra_cols if c not in in_fieldnames]

        with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()

            rows_written = 0
            for i, row in enumerate(reader):
                if i < args.start_row:
                    continue
                if args.limit and rows_written >= args.limit:
                    break

                user_message = row.get(prompt_col, "")
                user_message = "" if user_message is None else str(user_message)

                record = dict(row)
                record["error"] = ""

                try:
                    result = generate_one(
                        tokenizer=tokenizer,
                        model=model,
                        system_message=args.system_message,
                        user_message=user_message,
                        gen_kwargs=gen_kwargs,
                    )
                    record.update(result)
                except Exception as e:
                    record["completion_text"] = ""
                    record["prompt_tokens"] = ""
                    record["completion_tokens"] = ""
                    record["total_tokens"] = ""
                    record["generation_time_s"] = ""
                    record["error"] = f"{type(e).__name__}: {e}"

                writer.writerow(record)
                rows_written += 1

                if rows_written % args.flush_every == 0:
                    f_out.flush()

                if (i + 1) % 10 == 0:
                    print(f"[{model_choice}] processed input_row={i+1}, written={rows_written} -> {out_csv}")

                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

    print(f"[done] [{model_choice}] Wrote completions to: {out_csv}")
    print(f"[done] [{model_choice}] Wrote run metadata to: {out_json}")

    # Cleanup per model (important for ALL)
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["1B", "4B", "ALL"], default="1B")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", default="./runs")
    p.add_argument("--system_message", default=DEFAULT_SYSTEM_MESSAGE)

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--start_row", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--flush_every", type=int, default=1)
    p.add_argument("--sleep_ms", type=int, default=0)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # seeds
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model == "ALL":
        for m in ["1B", "4B"]:
            run_for_one_model(args, m)
    else:
        run_for_one_model(args, args.model)


if __name__ == "__main__":
    main()
