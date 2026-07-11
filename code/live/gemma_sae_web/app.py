# app.py
from flask import Flask, render_template, request, jsonify

from config import (
    DEFAULT_LAYER_INDEX,
    FLASK_DEBUG,
    GSM8K_BATCH_SIZE
)
from sae_backend import analyze_prompt_with_sae
from gsm8k_eval import evaluate_gsm8k, get_gsm8k_progress
from steering import (
    build_profile,
    register_custom_direction,
    register_teacher_forced_direction,
    list_available_profiles,
    parse_layer_range,
)
from model_backend import generate_batch_text_with_steering, build_metrics_for_text, get_tokenizer_and_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("gsm8k.html")


@app.route("/sae")
def sae_page():
    return render_template("sae.html", default_layer=DEFAULT_LAYER_INDEX)


@app.route("/gsm8k")
def gsm8k_page():
    return render_template("gsm8k.html")


@app.route("/api/sae_analyze", methods=["POST"])
def api_sae_analyze():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    layer_index = int(data.get("layer_index", DEFAULT_LAYER_INDEX))
    top_k = int(data.get("top_k", 20))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    result = analyze_prompt_with_sae(prompt, layer_index=layer_index, top_k_features=top_k)
    return jsonify(result)

@app.route("/api/gsm8k_eval", methods=["POST"])
def api_gsm8k_eval():
    data = request.get_json(force=True) or {}
    n_examples = int(data.get("n_examples", 50))
    batch_size = int(data.get("batch_size", 8))  # let you override

    steering_cfg = {
        "profile_name": data.get("steering_profile", "none"),
        "alpha": float(data.get("steering_alpha", 0.0)),
        "region": data.get("steering_region", "corridor"),
        "custom_range": data.get("steering_custom_range"),
        "scope": data.get("steering_scope", "completion"),
        "operator_aware": bool(data.get("steering_operator_aware", False)),
        "include_baseline": bool(data.get("include_baseline", False)),
    }
    if isinstance(steering_cfg["custom_range"], str):
        try:
            parts = [int(x) for x in steering_cfg["custom_range"].split("-")]
            steering_cfg["custom_range"] = parts if len(parts) == 2 else None
        except Exception:
            steering_cfg["custom_range"] = None

    result = evaluate_gsm8k(
        n_examples=n_examples,
        batch_size=batch_size,
        steering_config=steering_cfg,
    )
    return jsonify(result)


@app.route("/api/gsm8k_progress", methods=["GET"])
def api_gsm8k_progress():
    return jsonify(get_gsm8k_progress())


@app.route("/api/steer_generate", methods=["POST"])
def api_steer_generate():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    profile_name = data.get("profile", "none")
    alpha = float(data.get("alpha", 0.0))
    region = data.get("region", "corridor")
    scope = data.get("scope", "completion")
    custom_range = data.get("custom_range") or None
    if isinstance(custom_range, str):
        try:
            parts = [int(x) for x in custom_range.split("-")]
            custom_range = parts if len(parts) == 2 else None
        except Exception:
            custom_range = None

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    tok, model = get_tokenizer_and_model()
    total_layers = getattr(model.config, "num_hidden_layers", 0)
    profile = build_profile(
        profile_name,
        alpha=alpha,
        region_preset=region,
        scope=scope,
        total_layers=total_layers,
        custom_range=custom_range,
    )

    baseline = generate_batch_text_with_steering([prompt], profile=None)[0]
    steered = generate_batch_text_with_steering([prompt], profile=profile)[0] if profile else baseline
    hook_layers = []
    if profile:
        from steering import get_hook_indices_for_profile  # local import to avoid circular
        hook_layers = get_hook_indices_for_profile(profile, model)
    steering_applied = bool(profile) and alpha != 0.0 and bool(hook_layers)

    return jsonify({
        "prompt": prompt,
        "profile": profile_name,
        "alpha": alpha,
        "region": region,
        "scope": scope,
        "hook_layers": hook_layers,
        "num_hook_layers": len(hook_layers),
        "steering_applied": steering_applied,
        "baseline": baseline,
        "steered": steered,
        "baseline_metrics": build_metrics_for_text(baseline),
        "steered_metrics": build_metrics_for_text(steered),
    })


@app.route("/api/steering_profiles", methods=["GET", "POST"])
def api_steering_profiles():
    if request.method == "GET":
        return jsonify({"profiles": list_available_profiles()})

    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    mode = (data.get("mode") or "embedding").lower()
    scope = (data.get("scope") or "completion")
    layer_str = data.get("layers") or ""
    tok, model = get_tokenizer_and_model()
    total_layers = getattr(model.config, "num_hidden_layers", 0)
    layers = parse_layer_range(layer_str, total_layers)
    if not layers:
        # fallback to corridor heuristic if empty
        start = max(1, int(total_layers * 0.35))
        end = max(start, int(total_layers * 0.65))
        layers = list(range(start, end + 1))

    if mode == "teacher_forced":
        pos_raw = data.get("pos_examples") or []
        neg_raw = data.get("neg_examples") or []
        pos_examples = []
        neg_examples = []
        for ex in pos_raw:
            prompt = (ex.get("prompt") or "").strip()
            comp = (ex.get("completion") or "").strip()
            if prompt or comp:
                pos_examples.append((prompt, comp))
        for ex in neg_raw:
            prompt = (ex.get("prompt") or "").strip()
            comp = (ex.get("completion") or "").strip()
            if prompt or comp:
                neg_examples.append((prompt, comp))
        if not name or not pos_examples:
            return jsonify({"error": "name and at least one positive example are required"}), 400
        meta = register_teacher_forced_direction(
            name=name,
            pos_examples=pos_examples,
            neg_examples=neg_examples,
            layers=layers,
            scope=scope,
            tokenizer=tok,
            model=model,
        )
        return jsonify({"profile": meta, "profiles": list_available_profiles()})

    # default embedding-based
    pos = data.get("pos_prompts") or []
    neg = data.get("neg_prompts") or []
    if isinstance(pos, str):
        pos = [s.strip() for s in pos.split("\n") if s.strip()]
    if isinstance(neg, str):
        neg = [s.strip() for s in neg.split("\n") if s.strip()]
    if not name or not pos:
        return jsonify({"error": "name and at least one positive prompt are required"}), 400
    meta = register_custom_direction(name=name, pos_prompts=pos, neg_prompts=neg, tokenizer=tok, model=model)
    return jsonify({"profile": meta, "profiles": list_available_profiles()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=FLASK_DEBUG)

