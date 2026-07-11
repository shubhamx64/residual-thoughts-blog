import numpy as np
from pathlib import Path
import sys

def aggregate_phase(phase_num, file_pattern, output_filename):
    """
    Finds all .npz files matching a pattern, loads their arrays,
    prefixes keys with the filename stem, and saves to a single .npz.
    """
    # 1. Find files
    files = list(Path('.').glob(file_pattern))
    
    # Filter out the output file itself if it already exists and matches the pattern
    files = [f for f in files if f.name != output_filename]
    
    if not files:
        print(f"⚠️  Phase {phase_num}: No files found matching '{file_pattern}'")
        return

    print(f"📦 Phase {phase_num}: Aggregating {len(files)} files...")
    
    merged_data = {}
    
    # 2. Load and merge
    for f in files:
        try:
            # We use the filename stem (e.g., 'phase1_metrics_code') as a prefix
            # so that 'layer_indices' from 'code' doesn't overwrite 'stories'
            namespace = f.stem 
            
            with np.load(f) as data:
                for key in data.files:
                    # New key example: "phase1_metrics_code/mean_cos_mean"
                    new_key = f"{namespace}/{key}"
                    merged_data[new_key] = data[key]
                    
        except Exception as e:
            print(f"   ❌ Error loading {f.name}: {e}")

    # 3. Save aggregated file
    if merged_data:
        np.savez_compressed(output_filename, **merged_data)
        file_size_mb = Path(output_filename).stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved {output_filename} ({file_size_mb:.2f} MB)")
    else:
        print(f"   ⚠️  No data found to save for Phase {phase_num}.")

def main():
    print("Starting Aggregation...")
    
    # --- Phase 1: Token Clouds ---
    # Matches: phase1_metrics_code.npz, phase1_metrics_stories.npz, etc.
    aggregate_phase(1, "phase1_metrics_*.npz", "phase1_aggregated.npz")

    # --- Phase 2: Transport ---
    # Matches: google_gemma...phase2_transport_code.npz, etc.
    # We look for *phase2_transport* to ignore model name prefixes
    aggregate_phase(2, "*phase2_transport_*.npz", "phase2_aggregated.npz")

    # --- Phase 3: Layer Updates ---
    # Matches: phase3_metrics_code.npz, etc.
    aggregate_phase(3, "phase3_metrics_*.npz", "phase3_aggregated.npz")

    # --- Phase 4: Trajectories ---
    # Matches: phase4_metrics_code_last_token_all.npz, phase4_metrics_gsm8k...correct.npz, etc.
    aggregate_phase(4, "phase4_metrics_*.npz", "phase4_aggregated.npz")

    print("\nDone. You should now have 4 consolidated .npz files.")

if __name__ == "__main__":
    main()