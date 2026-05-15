"""
utils.py — Small helper functions shared across models.
"""

import os
import random
import numpy as np


def set_seed(seed=42):
    """Set random seed everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    print(f"Seed set to {seed}")


def make_dir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
