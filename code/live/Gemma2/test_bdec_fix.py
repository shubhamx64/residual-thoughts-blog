"""
Test: Does adding b_dec (decoder bias) fix the sign flip?
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

from activation_validation import create_attention_hooks, ActivationCache
from config import GEMMA2_CONFIG

TEST_LAYERS = [6, 14, 20, 24]
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test.",
    "Machine learning models process data through neural networks with layers."
]

device = "cuda"
cache = ActivationCache()

print("\n" + "="*70)
print("TEST: Does adding b_dec fix the sign flip?")
print("="*70)
print("Decoding WITH b_dec: x_recon = features @ W_dec.T + b_dec")
print("Decoding WITHOUT b_dec: x_recon = features @ W_dec.T")
print("="*70)

for test_layer in TEST_LAYERS:
    print(f"\n=== Layer {test_layer} ===")
    
    sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(test_layer)
    sae_id = f"layer_{sae_layer}/width_16k/canonical"
    sae = SAE.from_pretrained('gemma-scope-2b-pt-res-canonical', sae_id, device='cuda')[0]
    
    W_dec = sae.W_dec.data.float()  # [n_features, d_model]
    b_dec = sae.b_dec.data.float()  # [d_model]
    
    print(f"  b_dec norm: {b_dec.norm().item():.2f}")
    
    cos_without_bias = []
    cos_with_bias = []
    
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
            
            # Ground truth
            x_attn_true = attn_weights @ residual_pre
            
            # Encode
            flat = residual_pre.reshape(-1, residual_pre.shape[-1])
            features = sae.encode(flat.to(sae.W_enc.dtype)).float()
            features = features.reshape(seq_len, -1)
            
            # Attend in feature space
            attended_features = attn_weights @ features
            
            # Decode WITHOUT bias
            x_without_bias = attended_features @ W_dec
            
            # Decode WITH bias (correct way)
            x_with_bias = attended_features @ W_dec + b_dec
            
            for t in range(1, seq_len):
                true_vec = x_attn_true[t]
                
                if torch.norm(true_vec) > 1e-9:
                    # Without bias
                    if torch.norm(x_without_bias[t]) > 1e-9:
                        cos1 = F.cosine_similarity(true_vec.unsqueeze(0), x_without_bias[t].unsqueeze(0)).item()
                        cos_without_bias.append(cos1)
                    
                    # With bias
                    if torch.norm(x_with_bias[t]) > 1e-9:
                        cos2 = F.cosine_similarity(true_vec.unsqueeze(0), x_with_bias[t].unsqueeze(0)).item()
                        cos_with_bias.append(cos2)
    
    mean_without = np.mean(cos_without_bias) if cos_without_bias else 0.0
    mean_with = np.mean(cos_with_bias) if cos_with_bias else 0.0
    
    status_without = "✓" if mean_without > 0.5 else ("⚠️" if mean_without > 0 else "❌")
    status_with = "✓" if mean_with > 0.5 else ("⚠️" if mean_with > 0 else "❌")
    
    print(f"  Without b_dec: cos={mean_without:+.4f} {status_without}")
    print(f"  With b_dec:    cos={mean_with:+.4f} {status_with}")
    
    if mean_with > mean_without + 0.2:
        print(f"  → Adding b_dec HELPS! (+{mean_with - mean_without:.3f})")
    elif mean_with > 0 and mean_without < 0:
        print(f"  → b_dec FIXES the sign flip!")

print("\n" + "="*70)
