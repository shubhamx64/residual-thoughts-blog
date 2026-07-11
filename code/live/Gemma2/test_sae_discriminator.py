"""
Precise discriminator test for SAE mixing vs reconstruction issue.

Compares against ground truth x_attn_true:
- x_attn_sae = decode(attn @ encode(x))  [current: SAE-then-attend]
- x_attn_alt = decode(encode(x_attn_true))  [attend-then-SAE]

If x_attn_alt is fine but x_attn_sae flips → non-commutation is culprit
If BOTH flip → SAE missing structure (bias/mean/subspace) is culprit
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

from sae_utils import SAEManager
from activation_validation import create_attention_hooks, ActivationCache
from config import GEMMA2_CONFIG

TEST_LAYERS = [6, 14, 20, 24]
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test.",
    "Machine learning models process data through neural networks with layers."
]

device = "cuda"
cache = ActivationCache()
sae_manager = SAEManager(config=GEMMA2_CONFIG)

print("\n" + "="*70)
print("DISCRIMINATOR: Non-commutation vs Missing SAE structure")
print("="*70)
print("x_attn_true = attn @ residual (ground truth)")
print("x_attn_sae  = decode(attn @ encode(residual)) [SAE-then-attend]")
print("x_attn_alt  = decode(encode(x_attn_true)) [attend-then-SAE]")
print("\nIf alt is fine but sae flips → non-commutation issue")
print("If BOTH flip → SAE missing bias/mean/subspace")
print("="*70)

for test_layer in TEST_LAYERS:
    print(f"\n=== Layer {test_layer} ===")
    
    sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(test_layer)
    sae_decoder = sae_manager.get_decoder(sae_layer).cuda().float()
    
    cos_true_vs_sae = []  # cos(x_attn_true, x_attn_sae)
    cos_true_vs_alt = []  # cos(x_attn_true, x_attn_alt)
    
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs['input_ids'].shape[1]
        
        cache.clear()
        handles = create_attention_hooks(model, cache, [test_layer])
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        residual_pre = cache.residual_pre_attn.get(test_layer)
        if residual_pre is None:
            continue
        residual_pre = residual_pre[0].float()
        
        for h in handles:
            h.remove()
        
        for head_idx in range(8):
            attn_weights = outputs.attentions[test_layer][0, head_idx].float()
            
            # Ground truth: attend in residual space
            x_attn_true = attn_weights @ residual_pre  # [seq_len, d_model]
            
            # Method 1 (current): SAE-then-attend
            features_pre = sae_manager.encode(residual_pre.unsqueeze(0), sae_layer)[0].float()
            attended_features = attn_weights @ features_pre
            x_attn_sae = attended_features @ sae_decoder.T  # decode
            
            # Method 2 (alternative): attend-then-SAE
            features_attn = sae_manager.encode(x_attn_true.unsqueeze(0), sae_layer)[0].float()
            x_attn_alt = features_attn @ sae_decoder.T  # decode
            
            # Compare at each position (skip pos 0)
            for t in range(1, seq_len):
                true_vec = x_attn_true[t]
                sae_vec = x_attn_sae[t]
                alt_vec = x_attn_alt[t]
                
                if torch.norm(true_vec) > 1e-9:
                    if torch.norm(sae_vec) > 1e-9:
                        cos1 = F.cosine_similarity(true_vec.unsqueeze(0), sae_vec.unsqueeze(0)).item()
                        cos_true_vs_sae.append(cos1)
                    if torch.norm(alt_vec) > 1e-9:
                        cos2 = F.cosine_similarity(true_vec.unsqueeze(0), alt_vec.unsqueeze(0)).item()
                        cos_true_vs_alt.append(cos2)
    
    mean_sae = np.mean(cos_true_vs_sae) if cos_true_vs_sae else 0.0
    mean_alt = np.mean(cos_true_vs_alt) if cos_true_vs_alt else 0.0
    
    # Determine status
    sae_status = "✓" if mean_sae > 0.5 else ("⚠️" if mean_sae > 0 else "❌ FLIP")
    alt_status = "✓" if mean_alt > 0.5 else ("⚠️" if mean_alt > 0 else "❌ FLIP")
    
    print(f"  cos(true, sae-then-attend): {mean_sae:+.4f} {sae_status}")
    print(f"  cos(true, attend-then-sae): {mean_alt:+.4f} {alt_status}")
    
    if mean_alt > 0.5 and mean_sae < 0:
        print(f"  → Non-commutation is the issue!")
    elif mean_alt < 0 and mean_sae < 0:
        print(f"  → SAE missing structure (both fail)")
    elif mean_sae < mean_alt - 0.1:
        print(f"  → SAE-then-attend introduces ~{(mean_alt-mean_sae):.2f} extra error")

print("\n" + "="*70)
