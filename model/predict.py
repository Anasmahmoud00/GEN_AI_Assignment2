"""
model/predict.py
----------------
Inference script — required by the assignment.

Usage:
    python model/predict.py -i data/example_input.txt -o predictions.txt
    python model/predict.py -i data/example_input.txt -o predictions.txt --model gan
    python model/predict.py -i data/example_input.txt -o predictions.txt --model vae
    python model/predict.py -i data/example_input.txt -o predictions.txt --model mlp
    python model/predict.py -i data/example_input.txt -o predictions.txt --model flow

Input format  (example_input.txt):
    [WED] [JAN] [False] [196]
    [THU] [DEC] [True]  [204]
    ...

Output format (matches data.txt):
    [WED] [JAN] [False] [196] 3-1-1960
    [THU] [DEC] [True]  [204] 3-12-2048
    ...
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset  import load_conditions_only
from src.tokenizer import decode_date


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_gan(weights_dir="model/weights/gan"):
    from model.gan.model import build_generator, NOISE_DIM
    gen = build_generator()
    # build weights by calling once
    dummy = tf.zeros((1, NOISE_DIM + 22))
    gen(dummy)
    gen.load_weights(os.path.join(weights_dir, "generator.weights.h5"))
    print("Loaded GAN generator")
    return gen, "gan"


def load_vae(weights_dir="model/weights/vae"):
    from model.vae.model import build_decoder, LATENT_DIM
    dec = build_decoder()
    dummy = tf.zeros((1, LATENT_DIM + 22))
    dec(dummy)
    dec.load_weights(os.path.join(weights_dir, "decoder.weights.h5"))
    print("Loaded VAE decoder")
    return dec, "vae"


def load_mlp(weights_dir="model/weights/mlp"):
    from model.mlp.model import build_mlp
    model = build_mlp()
    dummy = tf.zeros((1, 22))
    model(dummy)
    model.load_weights(os.path.join(weights_dir, "mlp.weights.h5"))
    print("Loaded MLP")
    return model, "mlp"


def load_flow(weights_dir="model/weights/flow"):
    from model.flow.model import ConditionalRealNVP
    flow = ConditionalRealNVP()
    flow.load_weights(weights_dir)
    print("Loaded Normalizing Flow")
    return flow, "flow"


# ── Generate dates ─────────────────────────────────────────────────────────────

def generate(model, model_type, conditions_matrix):
    """
    Run inference for all condition rows.

    Returns list of date strings ["d-m-yyyy", ...]
    """
    n    = len(conditions_matrix)
    cond = tf.constant(conditions_matrix)

    if model_type == "gan":
        from model.gan.model import NOISE_DIM
        noise  = tf.random.normal([n, NOISE_DIM])
        inputs = tf.concat([noise, cond], axis=1)
        output = model(inputs, training=False).numpy()

    elif model_type == "vae":
        from model.vae.model import LATENT_DIM
        z      = tf.random.normal([n, LATENT_DIM])
        inputs = tf.concat([z, cond], axis=1)
        output = model(inputs, training=False).numpy()

    elif model_type == "mlp":
        output = model(cond, training=False).numpy()

    elif model_type == "flow":
        z      = tf.random.normal([n, 3])
        output = model.inverse(z, cond).numpy()
        # clamp to [0, 1] — flow output is unbounded
        output = np.clip(output, 0.0, 1.0)

    # decode each output vector → "d-m-yyyy"
    dates = [decode_date(output[i]) for i in range(n)]
    return dates


# ── Write output file ──────────────────────────────────────────────────────────

def write_output(raw_lines, dates, output_path):
    """
    Write predictions to output file.

    Each line = original condition string + generated date
    e.g.:  [WED] [JAN] [False] [196] 3-1-1960
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        for condition_str, date_str in zip(raw_lines, dates):
            f.write(f"{condition_str} {date_str}\n")

    print(f"Wrote {len(dates)} predictions → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dates Generator — inference script"
    )
    parser.add_argument("-i", required=True,
                        help="Path to input file (conditions only)")
    parser.add_argument("-o", required=True,
                        help="Path to output file (predictions)")
    parser.add_argument("--model", default="gan",
                        choices=["gan", "vae", "mlp", "flow"],
                        help="Which model to use (default: gan)")
    args = parser.parse_args()

    # load conditions
    X, raw_lines = load_conditions_only(args.i)
    print(f"Loaded {len(raw_lines)} conditions from {args.i}")

    # load model
    loaders = {
        "gan":  load_gan,
        "vae":  load_vae,
        "mlp":  load_mlp,
        "flow": load_flow,
    }
    model, model_type = loaders[args.model]()

    # generate
    dates = generate(model, model_type, X)

    # write output
    write_output(raw_lines, dates, args.o)


if __name__ == "__main__":
    main()
