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
    X = []
    Y = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = parse_line(line)

            if "y" not in row:
                continue

            year = row["y"]
            if year < 1800:
                continue
            if year > 2200:
                continue

            day_val = row["day"]
            month_val = row["month"]
            leap_val = row["leap"]
            decade_val = row["decade"]
            
            cond = encode_conditions(day_val, month_val, leap_val, decade_val)
            X.append(cond)
            
            d_val = row["d"]
            m_val = row["m"]
            y_val = row["y"]
            
            date_encoded = encode_date(d_val, m_val, y_val)
            Y.append(date_encoded)

    print(f"Loaded {len(X)} samples from {path}")

    random.seed(SEED)
    
    total_samples = len(X)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    X_shuffled = []
    Y_shuffled = []
    for i in indices:
        item_x = X[i]
        item_y = Y[i]
        X_shuffled.append(item_x)
        Y_shuffled.append(item_y)
    
    X = X_shuffled
    Y = Y_shuffled

    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[0 : n_train]
    Y_train = Y[0 : n_train]
    
    val_start = n_train
    val_end = n_train + n_val
    X_val = X[val_start : val_end]
    Y_val = Y[val_start : val_end]
    
    X_test = X[val_end : ]
    Y_test = Y[val_end : ]

    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    X_train_np = np.array(X_train, dtype=np.float32)
    Y_train_np = np.array(Y_train, dtype=np.float32)
    X_val_np   = np.array(X_val,   dtype=np.float32)
    Y_val_np   = np.array(Y_val,   dtype=np.float32)
    X_test_np  = np.array(X_test,  dtype=np.float32)
    Y_test_np  = np.array(Y_test,  dtype=np.float32)

    return X_train_np, Y_train_np, X_val_np, Y_val_np, X_test_np, Y_test_np


def load_conditions_only(path):
    from src.tokenizer import parse_line, encode_conditions

    X = []
    raw_lines = []

    with open(path, "r") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            
            row = parse_line(clean_line)
            
            day_val = row["day"]
            month_val = row["month"]
            leap_val = row["leap"]
            decade_val = row["decade"]
            
            cond = encode_conditions(day_val, month_val, leap_val, decade_val)
            X.append(cond)
            raw_lines.append(clean_line)

    X_np = np.array(X, dtype=np.float32)
    return X_np, raw_lines

