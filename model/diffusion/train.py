"""
model/diffusion/train.py
------------------------
Training loop for the Simplified Conditional Diffusion model.

Run:
    python model/diffusion/train.py

Saves weights to: model/weights/diffusion/diffusion.weights.h5
"""

import os
import sys
import numpy as np
import tensorflow as tf

root_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(root_path)

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_diffusion_model

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42
STEPS       = 100  # Number of diffusion steps

if os.path.exists("data/data.txt"):
    DATA_PATH = "data/data.txt"
else:
    DATA_PATH = "data.txt"

WEIGHTS_DIR = "model/weights/diffusion"

mse_loss_obj = tf.keras.losses.MeanSquaredError()


def get_noise_schedule(n_steps):
    # Linear schedule for beta (noise level)
    # We start with small noise and end with larger noise
    beta = np.linspace(0.0001, 0.02, n_steps)
    alpha = 1.0 - beta
    alpha_cumprod = np.cumprod(alpha)
    return tf.constant(alpha_cumprod, dtype=tf.float32)


alpha_cumprod = get_noise_schedule(STEPS)


def add_noise(real_dates, t_steps):
    # Formula: noisy = sqrt(alpha_cumprod) * real + sqrt(1 - alpha_cumprod) * noise
    
    # Get alpha values for the specific time steps in the batch
    # We need to reshape for broadcasting
    a = tf.gather(alpha_cumprod, t_steps)
    a = tf.reshape(a, [-1, 1])
    
    noise = tf.random.normal(tf.shape(real_dates))
    
    sqrt_a = tf.sqrt(a)
    sqrt_one_minus_a = tf.sqrt(1.0 - a)
    
    noisy_dates = sqrt_a * real_dates + sqrt_one_minus_a * noise
    return noisy_dates, noise


@tf.function
def train_step(conditions, real_dates, model, optimizer):
    batch_size = tf.shape(real_dates)[0]
    
    # Pick random time steps for each item in the batch
    t_steps = tf.random.uniform([batch_size], 0, STEPS, dtype=tf.int32)
    
    # Add noise to real dates according to the time steps
    noisy_dates, target_noise = add_noise(real_dates, t_steps)
    
    # Scale time steps to [0, 1] for better model performance
    t_scaled = tf.cast(t_steps, tf.float32) / float(STEPS)
    t_scaled = tf.reshape(t_scaled, [-1, 1])

    with tf.GradientTape() as tape:
        # Predict the noise that was added
        predicted_noise = model([noisy_dates, conditions, t_scaled], training=True)
        loss = mse_loss_obj(target_noise, predicted_noise)

    train_vars = model.trainable_variables
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))
    
    return loss


def train():
    set_seed(SEED)

    data_all = load_data(DATA_PATH)
    X_train = data_all[0]
    Y_train = data_all[1]
    X_val   = data_all[2]
    Y_val   = data_all[3]
    
    print(f"Training Diffusion on {len(X_train)} samples ...\n")

    model = build_diffusion_model()
    optimizer = tf.keras.optimizers.Adam(LR)

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            end_idx = i + BATCH_SIZE
            cond_batch = tf.constant(X_train[i : end_idx])
            date_batch = tf.constant(Y_train[i : end_idx])

            loss_val = train_step(cond_batch, date_batch, model, optimizer)
            epoch_loss = epoch_loss + float(loss_val)
            n_batches = n_batches + 1

        epoch_loss = epoch_loss / n_batches
        
        # Validation loss (simplified)
        val_t = tf.random.uniform([len(X_val)], 0, STEPS, dtype=tf.int32)
        val_noisy, val_noise_target = add_noise(tf.constant(Y_val), val_t)
        val_t_scaled = tf.cast(val_t, tf.float32) / float(STEPS)
        val_t_scaled = tf.reshape(val_t_scaled, [-1, 1])
        
        val_preds = model([val_noisy, tf.constant(X_val), val_t_scaled], training=False)
        val_loss = float(mse_loss_obj(val_noise_target, val_preds))

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  Train MSE: {epoch_loss:.4f}  Val MSE: {val_loss:.4f}")

    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    save_path = os.path.join(WEIGHTS_DIR, "diffusion.weights.h5")
    model.save_weights(save_path)
    
    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="Diffusion",
        save_path="plots/diffusion_loss.png",
    )

    return model


if __name__ == "__main__":
    train()
