"""
dataset.py — Load data.txt, encode everything, split into train/val/test.

Usage:
    from src.dataset import load_data

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data("data/data.txt")
"""

import random
import numpy as np
from src.tokenizer import parse_line, encode_conditions, encode_date

SEED = 42


def load_data(path, train_ratio=0.80, val_ratio=0.10):
    """
    Read data.txt and return train / val / test numpy arrays.

    Each line in data.txt looks like:
        [WED] [JAN] [False] [196] 3-12-1962

    Returns
    -------
    X_train, Y_train  — condition vectors (N, 22) and date vectors (N, 3)
    X_val,   Y_val
    X_test,  Y_test
    """
    X, Y = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = parse_line(line)

            # skip if no date (shouldn't happen in data.txt)
            if "y" not in row:
                continue

            # date range filter  [1800 – 2200]
            if not (1800 <= row["y"] <= 2200):
                continue

            X.append(encode_conditions(row["day"], row["month"], row["leap"], row["decade"]))
            Y.append(encode_date(row["d"], row["m"], row["y"]))

    print(f"Loaded {len(X)} samples from {path}")

    # shuffle before splitting
    random.seed(SEED)
    indices = list(range(len(X)))
    random.shuffle(indices)
    X = [X[i] for i in indices]
    Y = [Y[i] for i in indices]

    # split
    n        = len(X)
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)

    X_train, Y_train = X[:n_train],           Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],     Y[n_train+n_val:]

    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # convert to numpy
    return (
        np.array(X_train, dtype=np.float32), np.array(Y_train, dtype=np.float32),
        np.array(X_val,   dtype=np.float32), np.array(Y_val,   dtype=np.float32),
        np.array(X_test,  dtype=np.float32), np.array(Y_test,  dtype=np.float32),
    )


def load_conditions_only(path):
    """
    Load example_input.txt (conditions only, no date).

    Returns
    -------
    X          — numpy array (N, 22)
    raw_lines  — original strings, used to build the output file
    """
    from src.tokenizer import parse_line, encode_conditions

    X, raw_lines = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = parse_line(line)
            X.append(encode_conditions(row["day"], row["month"], row["leap"], row["decade"]))
            raw_lines.append(line)

    return np.array(X, dtype=np.float32), raw_lines
