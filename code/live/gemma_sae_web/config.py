# config.py
import os
import torch

# Core model
MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-1b-it")

# SAE repo & layer mapping (adjust to your actual Matryoshka release)
SAE_REPO_ID = os.environ.get(
    "SAE_REPO_ID",
    "gemma-3-1b-res-matryoshka-dc"  # replace with actual repo if different
)

# Which residual layer index to use for basic SAE demo
# This index is in "hidden_states" ordering (embedding = 0, first block = 1, etc.)
DEFAULT_LAYER_INDEX = int(os.environ.get("DEFAULT_LAYER_INDEX", "14"))

# Max tokens to generate
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

# Device / dtype
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Random seeds for reproducibility
RNG_SEED = int(os.environ.get("RNG_SEED", "42"))

# GSM8K settings
GSM8K_DATASET = "thesven/gsm8k-reasoning"
GSM8K_SPLIT = "train"  # this dataset only has 'train'
GSM8K_MAX_EXAMPLES = int(os.environ.get("GSM8K_MAX_EXAMPLES", "200"))

# Default batch size for GSM8K evaluation (you can override via API)
GSM8K_BATCH_SIZE = int(os.environ.get("GSM8K_BATCH_SIZE", "8"))

# Flask settings
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
