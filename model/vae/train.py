"""
model/vae/train.py
------------------
Training loop for the Conditional VAE.

Run:
    python model/vae/train.py

Saves weights to:  model/weights/vae/encoder.weights.h5
                   model/weights/vae/decoder.weights.h5

VAE Loss = Reconstruction Loss + KL Divergence
  - Reconstruction loss: how close is the decoded date to the real date?
  - KL divergence:       how close is the learned distribution to N(0,1)?
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_encoder, build_decoder, reparameterise, LATENT_DIM

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42
DATA_PATH   = "data/data.txt" if os.path.exists("data/data.txt") else "data.txt"
WEIGHTS_DIR = "model/weights/vae"


# ── VAE Loss ──────────────────────────────────────────────────────────────────

def vae_loss(real_date, reconstructed_date, mean, log_var):
    """
    Total VAE loss = reconstruction loss + KL divergence.

    Reconstruction loss
      MSE between real date and reconstructed date.
      (We use MSE because the output is continuous floats in [0,1])

    KL Divergence
      Measures how far the encoder distribution is from N(0,1).
      Formula: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    """
    # Reconstruction
    recon_loss = tf.reduce_mean(tf.square(real_date - reconstructed_date))

    # KL divergence
    kl_loss = -0.5 * tf.reduce_mean(
        1 + log_var - tf.square(mean) - tf.exp(log_var)
    )

    return recon_loss + kl_loss, recon_loss, kl_loss


# ── One training step ─────────────────────────────────────────────────────────

@tf.function
def train_step(conditions, real_dates, encoder, decoder, optimizer):
    """
    One batch update using tf.GradientTape (NO model.fit).

    Steps
    -----
    1. Encode (condition + real_date) → mean, log_var
    2. Sample z using reparameterisation trick
    3. Decode (z + condition) → reconstructed date
    4. Compute VAE loss and update both encoder + decoder
    """
    with tf.GradientTape() as tape:
        # Encode
        encoder_input = tf.concat([conditions, real_dates], axis=1)
        encoder_out   = encoder(encoder_input, training=True)

        mean    = encoder_out[:, :LATENT_DIM]
        log_var = encoder_out[:, LATENT_DIM:]

        # Sample latent z
        z = reparameterise(mean, log_var)

        # Decode
        decoder_input      = tf.concat([z, conditions], axis=1)
        reconstructed_date = decoder(decoder_input, training=True)

        # Loss
        loss, recon, kl = vae_loss(real_dates, reconstructed_date, mean, log_var)

    # Update both encoder and decoder together
    all_vars = encoder.trainable_variables + decoder.trainable_variables
    grads    = tape.gradient(loss, all_vars)
    optimizer.apply_gradients(zip(grads, all_vars))

    return loss, recon, kl


# ── Main training function ────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(DATA_PATH)
    print(f"Training VAE on {len(X_train)} samples …\n")

    encoder   = build_encoder()
    decoder   = build_decoder()
    encoder.summary()
    decoder.summary()

    optimizer = tf.keras.optimizers.Adam(LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            cond_batch = tf.constant(X_train[i : i + BATCH_SIZE])
            date_batch = tf.constant(Y_train[i : i + BATCH_SIZE])

            loss, recon, kl = train_step(
                cond_batch, date_batch, encoder, decoder, optimizer
            )
            epoch_loss += float(loss)
            n_batches  += 1

        epoch_loss /= n_batches

        # validation loss (no gradient)
        val_loss_total = 0.0
        for i in range(0, len(X_val), BATCH_SIZE):
            cond_val = tf.constant(X_val[i : i + BATCH_SIZE])
            date_val = tf.constant(Y_val[i : i + BATCH_SIZE])

            enc_out  = encoder(tf.concat([cond_val, date_val], axis=1), training=False)
            mean_v   = enc_out[:, :LATENT_DIM]
            logvar_v = enc_out[:, LATENT_DIM:]
            z_v      = reparameterise(mean_v, logvar_v)
            recon_v  = decoder(tf.concat([z_v, cond_val], axis=1), training=False)
            v_loss, _, _ = vae_loss(date_val, recon_v, mean_v, logvar_v)
            val_loss_total += float(v_loss)

        val_loss_total /= max(1, len(X_val) // BATCH_SIZE)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"Train: {epoch_loss:.4f}  Val: {val_loss_total:.4f}")

    # save weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    encoder.save_weights(os.path.join(WEIGHTS_DIR, "encoder.weights.h5"))
    decoder.save_weights(os.path.join(WEIGHTS_DIR, "decoder.weights.h5"))
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    # save loss plot
    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="VAE",
        save_path="plots/vae_loss.png",
    )

    return decoder   # decoder is what we use at inference


if __name__ == "__main__":
    train()
