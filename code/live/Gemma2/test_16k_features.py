"""
Quick test: OV_f validation with ALL 16K features on a late layer.
Tests whether the sign flip is due to feature subset sampling.
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from io import StringIO

# Suppress loading messages
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

# Import our modules
from sae_utils import SAEManager
from weight_extraction import extract_layer_weights
from activation_validation import create_attention_hooks, ActivationCache
from head_ablation import HeadAblator
from config import GEMMA2_CONFIG

# Test config
TEST_LAYER = 20  # Late layer with negative cosine
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
    "Machine learning models process data through neural networks with many layers."
]

print(f"\n=== Testing Layer {TEST_LAYER} with ALL 16K features ===\n")

# Load SAE
sae_manager = SAEManager(config=GEMMA2_CONFIG)
sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(TEST_LAYER)

# Get FULL decoder (all 16K features)
sae_decoder = sae_manager.get_decoder(sae_layer)  # [d_model, 16384]
print(f"Full decoder shape: {sae_decoder.shape}")

# No subsetting!
D = sae_decoder.cuda().float()
if GEMMA2_CONFIG.normalize_decoder_directions:
    D = F.normalize(D, dim=0) * (GEMMA2_CONFIG.hidden_size ** 0.5)

# Extract layer weights
layer_weights = extract_layer_weights(
    model, TEST_LAYER, GEMMA2_CONFIG,
    fold_gamma=False, device="cuda", dtype=torch.float32
)

device = "cuda"
cache = ActivationCache()

# Test each head
for head_idx in range(8):
    kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
    W_V = layer_weights.W_V[kv_group].cuda().float()
    W_O = layer_weights.W_O[head_idx].cuda().float()
    OV_circuit = W_V.T @ W_O.T
    
    all_cosine = []
    
    # Suppress ablator output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    ablator = HeadAblator(model, TEST_LAYER, head_idx)
    sys.stdout = old_stdout
    
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs['input_ids'].shape[1]
        
        # Baseline pass
        cache.clear()
        handles = create_attention_hooks(model, cache, [TEST_LAYER])
        with torch.no_grad():
            outputs_baseline = model(**inputs, output_attentions=True)
        
        residual_pre = cache.residual_pre_attn.get(TEST_LAYER)[0].float()
        attention_output_baseline = cache.attention_output.get(TEST_LAYER)[0].float()
        attn_weights = outputs_baseline.attentions[TEST_LAYER][0, head_idx].float()
        
        for h in handles:
            h.remove()
        
        # Ablated pass
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        ablator.ablate()
        sys.stdout = old_stdout
        
        cache.clear()
        handles = create_attention_hooks(model, cache, [TEST_LAYER])
        with torch.no_grad():
            model(**inputs)
        
        attention_output_ablated = cache.attention_output.get(TEST_LAYER)[0].float()
        
        for h in handles:
            h.remove()
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        ablator.restore()
        sys.stdout = old_stdout
        
        # Head-isolated output
        head_attention_output = attention_output_baseline - attention_output_ablated
        
        # Encode ALL features
        sae_pre = sae_manager.encode(residual_pre.unsqueeze(0), sae_layer)[0].float()
        
        # Vectorized prediction
        aggregated_features = attn_weights @ sae_pre  # [seq_len, 16384]
        attended_residuals = aggregated_features @ D.T  # [seq_len, d_model]
        pred_writes = attended_residuals @ OV_circuit
        
        pred_writes = pred_writes[1:]
        actual_writes = head_attention_output[1:]
        
        pred_norms = torch.norm(pred_writes, dim=1, keepdim=True)
        actual_norms = torch.norm(actual_writes, dim=1, keepdim=True)
        valid_mask = (pred_norms.squeeze() > 1e-9) & (actual_norms.squeeze() > 1e-9)
        
        if valid_mask.any():
            cos_sims = F.cosine_similarity(pred_writes[valid_mask], actual_writes[valid_mask], dim=1)
            all_cosine.extend(cos_sims.cpu().tolist())
    
    mean_cos = np.mean(all_cosine) if all_cosine else 0.0
    print(f"L{TEST_LAYER}H{head_idx}: Cosine = {mean_cos:+.4f} (n={len(all_cosine)})")

print("\n=== Compare with 4K subset result from report ===")
print("If signs match, feature subset is NOT the issue.")
print("If signs differ, feature subset IS the cause.")
