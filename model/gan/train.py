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

# ── allow imports from project root ──────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_generator, build_discriminator, NOISE_DIM

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.0002
SEED        = 42
DATA_PATH   = "data/data.txt" if os.path.exists("data/data.txt") else "data.txt"
WEIGHTS_DIR = "model/weights/gan"

# ── Loss function ─────────────────────────────────────────────────────────────
# Binary cross-entropy — standard GAN loss
bce = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    """
    D wants real → 1  and  fake → 0.
    Loss = bce(real, ones) + bce(fake, zeros)
    """
    real_loss = bce(tf.ones_like(real_output),  real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    """
    G wants D to think its output is real → label = 1.
    Loss = bce(fake, ones)
    """
    return bce(tf.ones_like(fake_output), fake_output)


# ── One training step ─────────────────────────────────────────────────────────

@tf.function   # compile to graph for speed
def train_step(conditions, real_dates, generator, discriminator,
               g_optimizer, d_optimizer):
    """
    One batch update using tf.GradientTape (NO model.fit).

    Steps
    -----
    1. Generate fake dates  G(noise, conditions)
    2. Train discriminator  on real + fake batch
    3. Train generator      to fool discriminator
    """
    batch_size = tf.shape(conditions)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

    # ── Discriminator update ──────────────────────────────────────────────
    with tf.GradientTape() as d_tape:
        fake_dates   = generator(tf.concat([noise, conditions], axis=1),
                                 training=True)

        real_input   = tf.concat([conditions, real_dates], axis=1)
        fake_input   = tf.concat([conditions, fake_dates], axis=1)

        real_output  = discriminator(real_input,  training=True)
        fake_output  = discriminator(fake_input,  training=True)

        d_loss       = discriminator_loss(real_output, fake_output)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # ── Generator update ──────────────────────────────────────────────────
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as g_tape:
        fake_dates  = generator(tf.concat([noise, conditions], axis=1),
                                training=True)
        fake_input  = tf.concat([conditions, fake_dates], axis=1)
        fake_output = discriminator(fake_input, training=False)

        g_loss      = generator_loss(fake_output)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss


# ── Main training function ────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    # load data  (only X_train / X_val used — Y is the real dates)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(DATA_PATH)
    print(f"Training GAN on {len(X_train)} samples …\n")

    # build models
    generator     = build_generator()
    discriminator = build_discriminator()
    generator.summary()
    discriminator.summary()

    # optimizers
    g_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5)

    # history for plotting
    g_losses, d_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        n_batches    = 0

        # manual batching (no tf.data needed)
        for i in range(0, len(X_train), BATCH_SIZE):
            cond_batch = tf.constant(X_train[i : i + BATCH_SIZE])
            date_batch = tf.constant(Y_train[i : i + BATCH_SIZE])

            d_loss, g_loss = train_step(
                cond_batch, date_batch,
                generator, discriminator,
                g_optimizer, d_optimizer,
            )
            d_epoch_loss += float(d_loss)
            g_epoch_loss += float(g_loss)
            n_batches    += 1

        d_epoch_loss /= n_batches
        g_epoch_loss /= n_batches

        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"D-loss: {d_epoch_loss:.4f}  "
                  f"G-loss: {g_epoch_loss:.4f}")

    # save weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    generator.save_weights(os.path.join(WEIGHTS_DIR, "generator.weights.h5"))
    discriminator.save_weights(os.path.join(WEIGHTS_DIR, "discriminator.weights.h5"))
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    # save loss plot
    plot_loss(
        train_losses=g_losses,
        val_losses=d_losses,
        model_name="GAN (G-loss vs D-loss)",
        save_path="plots/gan_loss.png",
    )

    return generator


if __name__ == "__main__":
    train()
