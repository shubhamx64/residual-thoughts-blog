import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

MODEL_ID = "google/gemma-3-1b-it"  # change if needed

def tree(mod, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    for name, child in mod.named_children():
        print(f"{prefix}{name}: {child.__class__.__name__}")
        tree(child, prefix + "  ", depth + 1, max_depth)

def try_get(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def describe_modulelist(name, ml):
    try:
        L = len(ml)
    except Exception:
        return
    print(f"\nFOUND ModuleList @ {name}  len={L}")
    if L > 0:
        print("  [0]:", ml[0].__class__.__name__)
        print("  [-1]:", ml[-1].__class__.__name__)
        # print common submodules inside first layer
        first = ml[0]
        print("  first-layer children:", [n for n,_ in first.named_children()])

def find_modulelists(mod, min_len=8):
    hits = []
    for name, child in mod.named_modules():
        if isinstance(child, torch.nn.ModuleList):
            try:
                L = len(child)
            except Exception:
                continue
            if L >= min_len:
                hits.append((name, child, L))
    hits.sort(key=lambda x: -x[2])
    return hits

print("Loading config:", MODEL_ID)
cfg = AutoConfig.from_pretrained(MODEL_ID)
print("Config class:", cfg.__class__.__name__)
print("Has text_config:", hasattr(cfg, "text_config"))
if hasattr(cfg, "text_config"):
    tc = cfg.text_config
    print("text_config:", tc.__class__.__name__)
    for k in ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim"]:
        if hasattr(tc, k):
            print(f"  text_config.{k} =", getattr(tc, k))

print("\nLoading AutoModelForCausalLM:", MODEL_ID)
m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map=None)
m.eval()
print("Top model class:", m.__class__.__name__)
print("Top-level children:", [n for n,_ in m.named_children()])

print("\n=== TREE (depth=2) ===")
tree(m, max_depth=2)

# Common candidates for "decoder layers"
candidates = [
    "model.layers",
    "model.model.layers",
    "model.decoder.layers",
    "model.text_model.layers",
    "model.language_model.layers",
    "model.transformer.layers",
    "model.gemma.layers",
    "model.text.layers",
    "layers",
]

print("\n=== PATH PROBES ===")
for p in candidates:
    x = try_get(m, p)
    if x is None:
        continue
    print(f"Path {p} -> {type(x).__name__}")
    if isinstance(x, torch.nn.ModuleList):
        describe_modulelist(p, x)

# Search for big ModuleLists anywhere
print("\n=== SEARCHING FOR LARGE ModuleLists (len>=8) ===")
hits = find_modulelists(m, min_len=8)
if not hits:
    print("No large ModuleLists found?!")
else:
    for name, ml, L in hits[:10]:
        describe_modulelist(name, ml)

# Also try loading AutoModel (sometimes the plain model has the layers)
print("\nLoading AutoModel (non-CausalLM):", MODEL_ID)
m2 = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map=None)
m2.eval()
print("AutoModel class:", m2.__class__.__name__)
print("AutoModel children:", [n for n,_ in m2.named_children()])

print("\n=== TREE AutoModel (depth=2) ===")
tree(m2, max_depth=2)

print("\n=== SEARCHING AutoModel for LARGE ModuleLists (len>=8) ===")
hits2 = find_modulelists(m2, min_len=8)
for name, ml, L in hits2[:10]:
    describe_modulelist(name, ml)
