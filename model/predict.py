"""
model/predict.py
----------------
Inference script — required by the assignment.

Usage:
    python model/predict.py -i data/example_input.txt -o predictions.txt
    python model/predict.py -i data/example_input.txt -o predictions.txt --model gan
    python model/predict.py -i data/example_input.txt -o predictions.txt --model vae
    python model/predict.py -i data/example_input.txt -o predictions.txt --model mlp
    python model/predict.py -i data/example_input.txt -o predictions.txt --model diffusion
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_path)

from src.dataset  import load_conditions_only
from src.tokenizer import decode_date


def load_gan(weights_dir="model/weights/gan"):
    from model.gan.model import build_generator, NOISE_DIM
    gen = build_generator()
    
    dummy_input = tf.zeros((1, NOISE_DIM + 22))
    gen(dummy_input)
    
    weights_path = os.path.join(weights_dir, "generator.weights.h5")
    gen.load_weights(weights_path)
    
    print("Loaded GAN generator")
    return gen, "gan"


def load_vae(weights_dir="model/weights/vae"):
    from model.vae.model import build_decoder, LATENT_DIM
    dec = build_decoder()
    
    dummy_input = tf.zeros((1, LATENT_DIM + 22))
    dec(dummy_input)
    
    weights_path = os.path.join(weights_dir, "decoder.weights.h5")
    dec.load_weights(weights_path)
    
    print("Loaded VAE decoder")
    return dec, "vae"


def load_mlp(weights_dir="model/weights/mlp"):
    from model.mlp.model import build_mlp
    model = build_mlp()
    
    dummy_input = tf.zeros((1, 22))
    model(dummy_input)
    
    weights_path = os.path.join(weights_dir, "mlp.weights.h5")
    model.load_weights(weights_path)
    
    print("Loaded MLP")
    return model, "mlp"


def load_diffusion(weights_dir="model/weights/diffusion"):
    from model.diffusion.model import build_diffusion_model
    model = build_diffusion_model()
    
    dummy_noisy = tf.zeros((1, 3))
    dummy_cond = tf.zeros((1, 22))
    dummy_time = tf.zeros((1, 1))
    model([dummy_noisy, dummy_cond, dummy_time])
    
    weights_path = os.path.join(weights_dir, "diffusion.weights.h5")
    model.load_weights(weights_path)
    
    print("Loaded Diffusion model")
    return model, "diffusion"


def generate(model, model_type, conditions_matrix):
    n = len(conditions_matrix)
    cond = tf.constant(conditions_matrix)

    if model_type == "gan":
        from model.gan.model import NOISE_DIM
        noise_shape = [n, NOISE_DIM]
        noise = tf.random.normal(noise_shape)
        
        inputs = tf.concat([noise, cond], axis=1)
        output_tf = model(inputs, training=False)
        output = output_tf.numpy()

    elif model_type == "vae":
        from model.vae.model import LATENT_DIM
        z_shape = [n, LATENT_DIM]
        z = tf.random.normal(z_shape)
        
        inputs = tf.concat([z, cond], axis=1)
        output_tf = model(inputs, training=False)
        output = output_tf.numpy()

    elif model_type == "mlp":
        output_tf = model(cond, training=False)
        output = output_tf.numpy()

    elif model_type == "diffusion":
        from model.diffusion.train import STEPS
        
        # Start with pure noise
        x = tf.random.normal([n, 3])
        
        # Number of sampling steps (can be fewer than training steps for speed)
        for t in range(STEPS - 1, -1, -1):
            t_val = tf.fill([n, 1], float(t) / float(STEPS))
            
            # Predict noise
            predicted_noise = model([x, cond, t_val], training=False)
            
            # Remove a bit of noise (Simplified update rule)
            # This is a very basic version of the DDPM sampling
            x = x - (0.01 * predicted_noise)
            
        output_raw = x.numpy()
        output = np.clip(output_raw, 0.0, 1.0)

    dates = []
    for i in range(n):
        row = output[i]
        date_str = decode_date(row)
        dates.append(date_str)
        
    return dates


def write_output(raw_lines, dates, output_path):
    abs_path = os.path.abspath(output_path)
    dir_name = os.path.dirname(abs_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w") as f:
        count = len(dates)
        for i in range(count):
            condition_str = raw_lines[i]
            date_str = dates[i]
            line_to_write = condition_str + " " + date_str + "\n"
            f.write(line_to_write)

    print(f"Wrote {len(dates)} predictions -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dates Generator")
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("--model", default="gan")
    args = parser.parse_args()

    input_path = args.i
    X, raw_lines = load_conditions_only(input_path)
    
    print(f"Loaded {len(raw_lines)} conditions from {input_path}")

    model_name = args.model
    if model_name == "gan":
        model, model_type = load_gan()
    elif model_name == "vae":
        model, model_type = load_vae()
    elif model_name == "mlp":
        model, model_type = load_mlp()
    elif model_name == "diffusion":
        model, model_type = load_diffusion()
    else:
        print("Unknown model type")
        return

    dates = generate(model, model_type, X)

    output_path = args.o
    write_output(raw_lines, dates, output_path)


if __name__ == "__main__":
    main()


