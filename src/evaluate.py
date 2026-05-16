"""
evaluate.py — Measure how well the model generates valid dates.

Main metric: Condition Satisfaction Rate (CSR)
  → What % of generated dates satisfy ALL 4 input conditions?
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from src.tokenizer import decode_date, check_date, parse_line


def evaluate(condition_lines, model_outputs):
    total = len(condition_lines)
    
    day_count = 0
    month_count = 0
    leap_count = 0
    decade_count = 0
    all_count = 0

    for i in range(total):
        line = condition_lines[i]
        conditions = parse_line(line)
        
        output_row = model_outputs[i]
        date_str = decode_date(output_row)
        
        result = check_date(date_str, conditions)

        if result["day_ok"]:
            day_count = day_count + 1
            
        if result["month_ok"]:
            month_count = month_count + 1
            
        if result["leap_ok"]:
            leap_count = leap_count + 1
            
        if result["decade_ok"]:
            decade_count = decade_count + 1
            
        if result["all_ok"]:
            all_count = all_count + 1

    csr_day = day_count / total
    csr_month = month_count / total
    csr_leap = leap_count / total
    csr_decade = decade_count / total
    csr_all = all_count / total

    metrics = {
        "csr_day":    csr_day,
        "csr_month":  csr_month,
        "csr_leap":   csr_leap,
        "csr_decade": csr_decade,
        "csr_all":    csr_all
    }
    return metrics


def print_metrics(metrics, model_name="Model"):
    print("\n-- " + model_name + " Results --")
    
    day_pct = metrics['csr_day'] * 100
    print(f"  Day match    : {day_pct:.1f}%")
    
    month_pct = metrics['csr_month'] * 100
    print(f"  Month match  : {month_pct:.1f}%")
    
    leap_pct = metrics['csr_leap'] * 100
    print(f"  Leap match   : {leap_pct:.1f}%")
    
    decade_pct = metrics['csr_decade'] * 100
    print(f"  Decade match : {decade_pct:.1f}%")
    
    all_pct = metrics['csr_all'] * 100
    print(f"  ALL pass     : {all_pct:.1f}%  (primary metric)")
    print("--------------------------------------\n")


def plot_loss(train_losses, val_losses, model_name, save_path):
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure(figsize=(8, 4))
    
    plt.plot(train_losses, label="Train Loss")
    
    if val_losses is not None:
        if len(val_losses) > 0:
            plt.plot(val_losses, label="Val Loss", linestyle="--")
            
    plt.title(model_name + " - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved plot: " + save_path)

