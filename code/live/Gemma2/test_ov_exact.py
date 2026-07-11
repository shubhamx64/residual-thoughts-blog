"""
No-SAE exact control for OV validation.

This sanity check uses RAW residuals (no SAE encoding/decoding) to verify:
1. Our OV_circuit = W_V.T @ W_O.T is correct
2. Our attention weight hooks capture the right values
3. Our ablation isolates the correct head contribution

If this gives cosine ≈ 1.0, the math is right and SAE is causing the sign flip.
If this gives cosine << 1.0, we have a hookpoint/weight/ablation bug.
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from io import StringIO

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

from weight_extraction import extract_layer_weights
from activation_validation import create_attention_hooks, ActivationCache
from head_ablation import HeadAblator
from config import GEMMA2_CONFIG

# Test configs
TEST_LAYERS = [6, 20]  # One early, one late layer
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process data through neural networks.",
]

device = "cuda"
cache = ActivationCache()

print("\n" + "="*70)
print("NO-SAE EXACT OV CONTROL")
print("Expected: cosine ≈ 1.0 if OV formula and hooks are correct")
print("="*70)

for test_layer in TEST_LAYERS:
    print(f"\n=== Layer {test_layer} ===")
    
    # Extract weights WITHOUT gamma (input is already post-RMSNorm)
    layer_weights = extract_layer_weights(
        model, test_layer, GEMMA2_CONFIG,
        fold_gamma=False, device="cuda", dtype=torch.float32
    )
    
    for head_idx in range(8):
        kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
        W_V = layer_weights.W_V[kv_group].cuda().float()  # [head_dim, d_model]
        W_O = layer_weights.W_O[head_idx].cuda().float()  # [d_model, head_dim]
        
        # OV circuit for row vectors: x @ W_V.T @ W_O.T
        OV_circuit = W_V.T @ W_O.T  # [d_model, d_model]
        
        all_cosine = []
        
        # Create ablator
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        ablator = HeadAblator(model, test_layer, head_idx)
        sys.stdout = old_stdout
        
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs['input_ids'].shape[1]
            
            # === Baseline pass ===
            cache.clear()
            handles = create_attention_hooks(model, cache, [test_layer])
            with torch.no_grad():
                outputs_baseline = model(**inputs, output_attentions=True)
            
            # Get raw residual (this is post-RMSNorm, pre-attention)
            residual_pre = cache.residual_pre_attn.get(test_layer)
            if residual_pre is None:
                print(f"  L{test_layer}H{head_idx}: residual_pre not captured!")
                continue
            residual_pre = residual_pre[0].float()  # [seq_len, d_model]
            
            attention_output_baseline = cache.attention_output.get(test_layer)
            if attention_output_baseline is None:
                print(f"  L{test_layer}H{head_idx}: attention_output not captured!")
                continue
            attention_output_baseline = attention_output_baseline[0].float()
            
            # Get attention weights for THIS head
            attn_weights = outputs_baseline.attentions[test_layer][0, head_idx].float()  # [seq_len, seq_len]
            
            for h in handles:
                h.remove()
            
            # === Ablated pass ===
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            ablator.ablate()
            sys.stdout = old_stdout
            
            cache.clear()
            handles = create_attention_hooks(model, cache, [test_layer])
            with torch.no_grad():
                model(**inputs)
            
            attention_output_ablated = cache.attention_output.get(test_layer)[0].float()
            
            for h in handles:
                h.remove()
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            ablator.restore()
            sys.stdout = old_stdout
            
            # === Head-isolated actual output ===
            head_actual = attention_output_baseline - attention_output_ablated  # [seq_len, d_model]
            
            # === NO-SAE prediction: use raw residual ===
            # attended = Σ_s α[t,s] * residual_pre[s]  (attention-weighted residual)
            attended_residual = attn_weights @ residual_pre  # [seq_len, d_model]
            
            # Apply OV circuit
            head_predicted = attended_residual @ OV_circuit  # [seq_len, d_model]
            
            # Compare (skip position 0)
            pred = head_predicted[1:]
            actual = head_actual[1:]
            
            pred_norms = torch.norm(pred, dim=1, keepdim=True)
            actual_norms = torch.norm(actual, dim=1, keepdim=True)
            valid_mask = (pred_norms.squeeze() > 1e-9) & (actual_norms.squeeze() > 1e-9)
            
            if valid_mask.any():
                cos_sims = F.cosine_similarity(pred[valid_mask], actual[valid_mask], dim=1)
                all_cosine.extend(cos_sims.cpu().tolist())
        
        mean_cos = np.mean(all_cosine) if all_cosine else 0.0
        std_cos = np.std(all_cosine) if all_cosine else 0.0
        
        # Flag unexpected results
        status = "✓" if mean_cos > 0.9 else "⚠️" if mean_cos > 0.5 else "❌"
        print(f"  L{test_layer}H{head_idx}: cos={mean_cos:+.4f} (std={std_cos:.4f}) {status}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("  cos ≈ 1.0: OV formula is correct, SAE causes the sign flip")
print("  cos << 1.0: Bug in hooks/weights/ablation, not SAE")
print("="*70)
