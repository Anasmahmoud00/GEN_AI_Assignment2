"""
utils.py — Small helper functions shared across models.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed set to {seed}")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

