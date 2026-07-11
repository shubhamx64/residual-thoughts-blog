#!/bin/bash

# ==============================================================================
# Experiment Pipeline Orchestrator
# ==============================================================================
# This script runs Phases 1-4 sequentially and consolidates the data.
# It does NOT run the meta-analysis automatically.
#
# Usage:
#   # Run pipeline and collect results into default folder 'final_analysis_data'
#   ./run_pipeline.sh
#
#   # Run pipeline and collect results into custom folder
#   ./run_pipeline.sh my_custom_results_dir
# ==============================================================================

set -e  # Exit immediately if a command fails

# ------------------------------------------------------------------------------
# 0. CONFIGURATION & PRE-FLIGHT CHECK
# ------------------------------------------------------------------------------
PYTHON_CMD="python" # Often just 'python' on Windows
RESULTS_DIR="${1:-final_analysis_data}" # Use arg 1 or default

SCRIPTS=(
    "phase1_token_cloud.py"
    "phase2_token_transport.py"
    "phase3_token_attn_mlp.py"
    "phase4_token_trajectory.py"
)

# Check if scripts exist
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "❌ Error: Could not find $script in the current directory."
        exit 1
    fi
done

echo "✅ All scripts found. Starting pipeline..."
echo "📂 Results will be collected in: $RESULTS_DIR"

# ------------------------------------------------------------------------------
# 1. EXECUTE PHASES
# ------------------------------------------------------------------------------

echo "----------------------------------------------------------------"
echo "🚀 STARTING PHASE 1: Token Clouds (Homogeneity Analysis)"
echo "----------------------------------------------------------------"
$PYTHON_CMD phase1_token_cloud.py

echo "----------------------------------------------------------------"
echo "🚀 STARTING PHASE 2: Information Transport (Lag Analysis)"
echo "----------------------------------------------------------------"
$PYTHON_CMD phase2_token_transport.py

echo "----------------------------------------------------------------"
echo "🚀 STARTING PHASE 3: Layer Updates (Attn vs MLP Norms)"
echo "----------------------------------------------------------------"
$PYTHON_CMD phase3_token_attn_mlp.py

echo "----------------------------------------------------------------"
echo "🚀 STARTING PHASE 4: Trajectories (Curvature & Geometry)"
echo "----------------------------------------------------------------"
$PYTHON_CMD phase4_token_trajectory.py


# ------------------------------------------------------------------------------
# 2. CONSOLIDATE DATA
# ------------------------------------------------------------------------------
# The phases output data into scattered subfolders.
# This section finds all .npz files and copies them to RESULTS_DIR.

echo "----------------------------------------------------------------"
echo "📦 CONSOLIDATING ARTIFACTS"
echo "----------------------------------------------------------------"

# Create a fresh results directory
if [ -d "$RESULTS_DIR" ]; then
    echo "   Cleaning existing results directory..."
    rm -rf "$RESULTS_DIR"
fi
mkdir -p "$RESULTS_DIR"

echo "   Gathering .npz files from phase output folders..."

# Find and copy all .npz files from any directory matching "phase*_outputs*"
# We use a loop to handle potential spaces in filenames, though unlikely here.
find . -type d -name "phase*_outputs_*" -print0 | while IFS= read -r -d '' dir; do
    find "$dir" -name "*.npz" -exec cp {} "$RESULTS_DIR/" \;
done

count=$(ls "$RESULTS_DIR"/*.npz 2>/dev/null | wc -l)
echo "✅ Copied $count .npz files to $RESULTS_DIR/"

echo "================================================================"
echo "🎉 PIPELINE COMPLETE"
echo "================================================================"
echo "Data collection finished. You can now run your analysis tools on:"
echo "   📂 $RESULTS_DIR"