"""
Test SAE nonlinearity / commutativity issue.

Compares:
A: x_attn = attn @ residual_pre, then encode+decode (correct order)
B: decode(attn @ encode(residual_pre)) (current order - mixing in feature space)

If A and B point opposite in late layers, the SAE nonlinearity is the culprit.
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

# Test layers
TEST_LAYERS = [6, 14, 20, 24]  # Early, mid, late, very late
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test.",
    "Machine learning models process data through neural networks with layers."
]

device = "cuda"
cache = ActivationCache()
sae_manager = SAEManager(config=GEMMA2_CONFIG)

print("\n" + "="*70)
print("SAE NONLINEARITY / COMMUTATIVITY TEST")
print("A = attend-then-SAE: encode(decode(attn @ residual))")
print("B = SAE-then-attend: decode(attn @ encode(residual))")
print("If cosine(A,B) < 0 in late layers, nonlinearity causes sign flip")
print("="*70)

for test_layer in TEST_LAYERS:
    print(f"\n=== Layer {test_layer} ===")
    
    sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(test_layer)
    sae_decoder = sae_manager.get_decoder(sae_layer).cuda().float()
    
    all_cosine_AB = []
    
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
        residual_pre = residual_pre[0].float()  # [seq_len, d_model]
        
        for h in handles:
            h.remove()
        
        # Get attention weights for each head and compare A vs B
        for head_idx in range(8):
            attn_weights = outputs.attentions[test_layer][0, head_idx].float()  # [seq_len, seq_len]
            
            # === Method A: Attend in residual space, then SAE ===
            # Step 1: Attend in residual space
            attended_residual_A = attn_weights @ residual_pre  # [seq_len, d_model]
            # Step 2: Encode+decode (this is for comparison, in real use we'd just use attended_residual_A)
            features_A = sae_manager.encode(attended_residual_A.unsqueeze(0), sae_layer)[0].float()
            reconstructed_A = features_A @ sae_decoder.T  # [seq_len, d_model]
            
            # === Method B: Current approach - SAE first, then attend ===
            # Step 1: Encode each position
            features_pre = sae_manager.encode(residual_pre.unsqueeze(0), sae_layer)[0].float()
            # Step 2: Attend in feature space
            attended_features_B = attn_weights @ features_pre  # [seq_len, n_features]
            # Step 3: Decode
            reconstructed_B = attended_features_B @ sae_decoder.T  # [seq_len, d_model]
            
            # Compare A vs B using cosine similarity at each position
            for t in range(1, seq_len):
                A_vec = reconstructed_A[t]
                B_vec = reconstructed_B[t]
                
                if torch.norm(A_vec) > 1e-9 and torch.norm(B_vec) > 1e-9:
                    cos_AB = F.cosine_similarity(A_vec.unsqueeze(0), B_vec.unsqueeze(0)).item()
                    all_cosine_AB.append(cos_AB)
    
    mean_cos = np.mean(all_cosine_AB) if all_cosine_AB else 0.0
    std_cos = np.std(all_cosine_AB) if all_cosine_AB else 0.0
    
    # Flag issues
    if mean_cos < 0:
        status = "❌ SIGN FLIP!"
    elif mean_cos < 0.5:
        status = "⚠️ Low agreement"
    else:
        status = "✓ Good"
    
    print(f"  cos(A,B) = {mean_cos:+.4f} (std={std_cos:.4f}) {status}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("  cos ≈ +1: SAE commutes well with attention (linearity preserved)")
print("  cos ≈ 0:  SAE distorts significantly (mixed directions)")
print("  cos < 0:  SAE causes sign flip (sparsity threshold artifacts)")
print("="*70)
