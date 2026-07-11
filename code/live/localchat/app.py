import torch
import gc
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread, Event

app = Flask(__name__)

# --- Global State ---
current_model = None
current_tokenizer = None
current_model_name = None
active_stop_event = None
active_thread = None

MODEL_MAP = {
    "1B": "google/gemma-3-1b-it", 
    "4B": "google/gemma-3-4b-it",
    "2B": "google/gemma-2-2b",
}

SYSTEM_MESSAGE = (
    "You are Gemma-3, a large language model trained by Google."
)

class StopOnEvent(StoppingCriteria):
    """Halts generation early when the stop event is set."""

    def __init__(self, stop_event: Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

class StopOnSequences(StoppingCriteria):
    """Stop generation when specified token sequences appear."""

    def __init__(self, tokenizer, sequences):
        super().__init__()
        self.sequence_ids = []
        for seq in sequences:
            try:
                tokenized = tokenizer(seq, add_special_tokens=False)["input_ids"]
            except Exception:
                continue
            if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
                tokenized = tokenized[0]
            if not tokenized:
                continue
            self.sequence_ids.append(tuple(tokenized))

    def __call__(self, input_ids, scores, **kwargs):
        if not self.sequence_ids or input_ids.size(-1) == 0:
            return False
        generated = input_ids[0].tolist()
        for seq in self.sequence_ids:
            seq_len = len(seq)
            if seq_len == 0 or seq_len > len(generated):
                continue
            if generated[-seq_len:] == list(seq):
                return True
        return False

def load_model_if_needed(choice):
    global current_model, current_tokenizer, current_model_name
    
    target_id = MODEL_MAP.get(choice)
    if not target_id:
        raise ValueError("Invalid model choice")

    # If the requested model is already loaded, do nothing
    if current_model_name == choice:
        return

    print(f"Loading {choice} ({target_id})...")
    
    # 1. Unload previous model to save VRAM
    if current_model is not None:
        del current_model
        del current_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # 2. Load new model
    try:
        current_tokenizer = AutoTokenizer.from_pretrained(target_id)
        current_model = AutoModelForCausalLM.from_pretrained(
            target_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16 # efficient for newer GPUs
        )
        if current_tokenizer.pad_token is None:
            current_tokenizer.pad_token = current_tokenizer.eos_token
        if current_model.config.pad_token_id is None and current_tokenizer.pad_token_id is not None:
            current_model.config.pad_token_id = current_tokenizer.pad_token_id
        current_model_name = choice
        print(f"Successfully loaded {choice}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def build_prompt_from_messages(messages):
    """Fallback prompt formatter for models without chat templates."""
    prompt_lines = []
    for message in messages:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "").strip()
        if not content:
            continue
        prompt_lines.append(f"{role}: {content}")
    prompt_lines.append("Assistant:")
    return "\n\n".join(prompt_lines)

def prepare_model_inputs(tokenizer, model, messages, raw_prompt=None):
    """Return dict of input tensors regardless of tokenizer chat template support."""
    if raw_prompt is not None:
        encoded = tokenizer(raw_prompt, return_tensors="pt")
        return {k: v.to(model.device) for k, v in encoded.items()}

    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    prompt_text = build_prompt_from_messages(messages)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    return encoded

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get('message', '')
    model_choice = payload.get('model') or "1B"
    max_tokens = payload.get('max_tokens', 512)
    temperature = payload.get('temperature', 0.7)
    raw_stop_sequences = payload.get('stop_sequences')
    if raw_stop_sequences is None:
        raw_stop_sequences = ["<END>"]

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 512
    max_tokens = max(1, min(4096, max_tokens))

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    temperature = max(0.0, min(2.0, temperature))

    stop_sequences = []
    if isinstance(raw_stop_sequences, str):
        candidates = [part.strip() for part in raw_stop_sequences.split(",")]
    elif isinstance(raw_stop_sequences, list):
        candidates = []
        for item in raw_stop_sequences:
            if isinstance(item, str):
                candidates.append(item.strip())
    else:
        candidates = []

    for candidate in candidates:
        if candidate:
            stop_sequences.append(candidate)

    global active_stop_event, active_thread

    # Ensure correct model is loaded
    try:
        load_model_if_needed(model_choice)
    except Exception as e:
        return Response(f"Error loading model: {str(e)}", status=500)

    # Stop any in-flight generation before starting a new one
    if active_stop_event:
        active_stop_event.set()
    if active_thread and active_thread.is_alive():
        active_thread.join(timeout=0.01)

    stop_event = Event()
    active_stop_event = stop_event

    # Prepare inputs
    chat_messages = []
    raw_prompt = None
    if model_choice == "2B":
        raw_prompt = user_input
    else:
        chat_messages.append({"role": "system", "content": SYSTEM_MESSAGE})
        chat_messages.append({"role": "user", "content": user_input})

    try:
        model_inputs = prepare_model_inputs(current_tokenizer, current_model, chat_messages, raw_prompt=raw_prompt)
    except Exception as e:
        return Response(f"Error preparing prompt: {str(e)}", status=500)

    # Streamer setup
    streamer = TextIteratorStreamer(current_tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping_criteria = [StopOnEvent(stop_event)]
    if stop_sequences:
        stopping_criteria.append(StopOnSequences(current_tokenizer, stop_sequences))
    
    # Generation args
    gen_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList(stopping_criteria),
    }

    # Run generation in a separate thread so we can stream the main thread
    thread = Thread(target=current_model.generate, kwargs=gen_kwargs)
    active_thread = thread
    thread.start()

    def generate():
        for new_text in streamer:
            yield new_text
        # Clear references after completion
        global active_stop_event, active_thread
        active_stop_event = None
        active_thread = None

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/stop', methods=['POST'])
def stop():
    """Signal any active generation to stop early."""
    global active_stop_event
    if active_stop_event:
        active_stop_event.set()
    return jsonify({"status": "stopping"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
