"""
evaluate.py — Measure how well the model generates valid dates.

Main metric: Condition Satisfaction Rate (CSR)
  → What % of generated dates satisfy ALL 4 input conditions?

(Accuracy is NOT used — this is a generative problem with many valid answers.)
"""

import matplotlib.pyplot as plt
from src.tokenizer import decode_date, check_date


def evaluate(condition_lines, model_outputs):
    """
    Check each generated date against its conditions.

    Parameters
    ----------
    condition_lines : list of raw strings  e.g. ["[WED] [JAN] [False] [196]", ...]
    model_outputs   : list/array of shape (N, 3)  — model output vectors

    Returns
    -------
    dict with CSR scores
    """
    from src.tokenizer import parse_line

    total = len(condition_lines)
    counts = {"day": 0, "month": 0, "leap": 0, "decade": 0, "all": 0}

    for i in range(total):
        conditions = parse_line(condition_lines[i])
        date_str   = decode_date(model_outputs[i])
        result     = check_date(date_str, conditions)

        if result["day_ok"]:    counts["day"]    += 1
        if result["month_ok"]:  counts["month"]  += 1
        if result["leap_ok"]:   counts["leap"]   += 1
        if result["decade_ok"]: counts["decade"] += 1
        if result["all_ok"]:    counts["all"]    += 1

    return {
        "csr_day":    counts["day"]    / total,
        "csr_month":  counts["month"]  / total,
        "csr_leap":   counts["leap"]   / total,
        "csr_decade": counts["decade"] / total,
        "csr_all":    counts["all"]    / total,   # ← main metric
    }


def print_metrics(metrics, model_name="Model"):
    """Print a simple summary table."""
    print(f"\n── {model_name} Results ──────────────")
    print(f"  Day match    : {metrics['csr_day']:.1%}")
    print(f"  Month match  : {metrics['csr_month']:.1%}")
    print(f"  Leap match   : {metrics['csr_leap']:.1%}")
    print(f"  Decade match : {metrics['csr_decade']:.1%}")
    print(f"  ALL pass     : {metrics['csr_all']:.1%}  ← primary metric")
    print(f"──────────────────────────────────────\n")


def plot_loss(train_losses, val_losses, model_name, save_path):
    """
    Save a loss curve plot as PNG.

    Parameters
    ----------
    train_losses : list of floats (one per epoch)
    val_losses   : list of floats (one per epoch) or None
    model_name   : str
    save_path    : str  e.g. "plots/gan_loss.png"
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Val Loss", linestyle="--")
    plt.title(f"{model_name} — Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")
