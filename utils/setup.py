"""
VisionSSL Common Setup Utilities
"""

from pathlib import Path
import random
import numpy as np
import torch


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/VisionSSL")

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

for directory in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Random Seed
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    print(f"Random Seed: {seed}")
