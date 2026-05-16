"""
model/gan/train.py
------------------
Training loop for the Conditional GAN.

Run:
    python model/gan/train.py

Saves weights to:  model/weights/gan/generator.weights.h5
                   model/weights/gan/discriminator.weights.h5
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
from model import build_generator, build_discriminator, NOISE_DIM

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.0002
SEED        = 42

if os.path.exists("data/data.txt"):
    DATA_PATH = "data/data.txt"
else:
    DATA_PATH = "data.txt"

WEIGHTS_DIR = "model/weights/gan"

bce_loss_obj = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    ones_labels = tf.ones_like(real_output)
    real_loss = bce_loss_obj(ones_labels,  real_output)
    
    zeros_labels = tf.zeros_like(fake_output)
    fake_loss = bce_loss_obj(zeros_labels, fake_output)
    
    total_d_loss = real_loss + fake_loss
    return total_d_loss


def generator_loss(fake_output):
    ones_labels = tf.ones_like(fake_output)
    g_loss = bce_loss_obj(ones_labels, fake_output)
    return g_loss


@tf.function
def train_step(conditions, real_dates, generator, discriminator, g_optimizer, d_optimizer):
    batch_size = tf.shape(conditions)[0]
    noise_shape = [batch_size, NOISE_DIM]
    
    noise = tf.random.normal(noise_shape)

    with tf.GradientTape() as d_tape:
        gen_input = tf.concat([noise, conditions], axis=1)
        fake_dates = generator(gen_input, training=True)

        real_input = tf.concat([conditions, real_dates], axis=1)
        fake_input = tf.concat([conditions, fake_dates], axis=1)

        real_output = discriminator(real_input,  training=True)
        fake_output = discriminator(fake_input,  training=True)

        d_loss = discriminator_loss(real_output, fake_output)

    d_vars = discriminator.trainable_variables
    d_grads = d_tape.gradient(d_loss, d_vars)
    
    d_grads_vars = zip(d_grads, d_vars)
    d_optimizer.apply_gradients(d_grads_vars)

    noise_2 = tf.random.normal(noise_shape)

    with tf.GradientTape() as g_tape:
        gen_input_2 = tf.concat([noise_2, conditions], axis=1)
        fake_dates_2 = generator(gen_input_2, training=True)
        
        fake_input_2 = tf.concat([conditions, fake_dates_2], axis=1)
        fake_output_2 = discriminator(fake_input_2, training=False)

        g_loss = generator_loss(fake_output_2)

    g_vars = generator.trainable_variables
    g_grads = g_tape.gradient(g_loss, g_vars)
    
    g_grads_vars = zip(g_grads, g_vars)
    g_optimizer.apply_gradients(g_grads_vars)

    return d_loss, g_loss


def train():
    set_seed(SEED)

    data_all = load_data(DATA_PATH)
    X_train = data_all[0]
    Y_train = data_all[1]
    X_val   = data_all[2]
    Y_val   = data_all[3]
    X_test  = data_all[4]
    Y_test  = data_all[5]
    
    print(f"Training GAN on {len(X_train)} samples ...\n")

    generator     = build_generator()
    discriminator = build_discriminator()
    generator.summary()
    discriminator.summary()

    g_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)

    g_losses = []
    d_losses = []

    for epoch in range(1, EPOCHS + 1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        n_batches    = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            end_idx = i + BATCH_SIZE
            cond_batch = tf.constant(X_train[i : end_idx])
            date_batch = tf.constant(Y_train[i : end_idx])

            losses = train_step(
                cond_batch, date_batch,
                generator, discriminator,
                g_optimizer, d_optimizer,
            )
            d_loss_val, g_loss_val = losses
            
            d_epoch_loss = d_epoch_loss + float(d_loss_val)
            g_epoch_loss = g_epoch_loss + float(g_loss_val)
            n_batches = n_batches + 1

        d_epoch_loss = d_epoch_loss / n_batches
        g_epoch_loss = g_epoch_loss / n_batches

        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  D-loss: {d_epoch_loss:.4f}  G-loss: {g_epoch_loss:.4f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    gen_path = os.path.join(WEIGHTS_DIR, "generator.weights.h5")
    disc_path = os.path.join(WEIGHTS_DIR, "discriminator.weights.h5")
    
    generator.save_weights(gen_path)
    discriminator.save_weights(disc_path)
    
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    plot_loss(
        train_losses=g_losses,
        val_losses=d_losses,
        model_name="GAN (G-loss vs D-loss)",
        save_path="plots/gan_loss.png",
    )

    return generator


if __name__ == "__main__":
    train()

