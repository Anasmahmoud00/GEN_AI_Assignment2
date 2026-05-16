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

project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(project_root)

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_encoder, build_decoder, reparameterise, LATENT_DIM

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42

if os.path.exists("data/data.txt"):
    DATA_PATH = "data/data.txt"
else:
    DATA_PATH = "data.txt"

WEIGHTS_DIR = "model/weights/vae"


def vae_loss(real_date, reconstructed_date, mean, log_var):
    diff = real_date - reconstructed_date
    square_diff = tf.square(diff)
    recon_loss = tf.reduce_mean(square_diff)

    term1 = 1 + log_var
    term2 = tf.square(mean)
    term3 = tf.exp(log_var)
    
    kl_elements = term1 - term2 - term3
    kl_loss = -0.5 * tf.reduce_mean(kl_elements)

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


@tf.function
def train_step(conditions, real_dates, encoder, decoder, optimizer):
    with tf.GradientTape() as tape:
        encoder_input = tf.concat([conditions, real_dates], axis=1)
        encoder_out   = encoder(encoder_input, training=True)

        mean    = encoder_out[:, :LATENT_DIM]
        log_var = encoder_out[:, LATENT_DIM:]

        z = reparameterise(mean, log_var)

        decoder_input      = tf.concat([z, conditions], axis=1)
        reconstructed_date = decoder(decoder_input, training=True)

        loss_tuple = vae_loss(real_dates, reconstructed_date, mean, log_var)
        loss, recon, kl = loss_tuple

    all_vars = encoder.trainable_variables + decoder.trainable_variables
    grads    = tape.gradient(loss, all_vars)
    
    zip_grads_vars = zip(grads, all_vars)
    optimizer.apply_gradients(zip_grads_vars)

    return loss, recon, kl


def train():
    set_seed(SEED)

    data_tuple = load_data(DATA_PATH)
    X_train = data_tuple[0]
    Y_train = data_tuple[1]
    X_val   = data_tuple[2]
    Y_val   = data_tuple[3]
    X_test  = data_tuple[4]
    Y_test  = data_tuple[5]
    
    print(f"Training VAE on {len(X_train)} samples ...\n")

    encoder   = build_encoder()
    decoder   = build_decoder()
    encoder.summary()
    decoder.summary()

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

            loss, recon, kl = train_step(
                cond_batch, date_batch, encoder, decoder, optimizer
            )
            epoch_loss = epoch_loss + float(loss)
            n_batches  = n_batches + 1

        epoch_loss = epoch_loss / n_batches

        val_loss_total = 0.0
        n_val_batches = 0
        for i in range(0, len(X_val), BATCH_SIZE):
            end_idx_val = i + BATCH_SIZE
            cond_val = tf.constant(X_val[i : end_idx_val])
            date_val = tf.constant(Y_val[i : end_idx_val])

            val_input = tf.concat([cond_val, date_val], axis=1)
            enc_out  = encoder(val_input, training=False)
            
            mean_v   = enc_out[:, :LATENT_DIM]
            logvar_v = enc_out[:, LATENT_DIM:]
            z_v      = reparameterise(mean_v, logvar_v)
            
            dec_input = tf.concat([z_v, cond_val], axis=1)
            recon_v  = decoder(dec_input, training=False)
            
            v_loss_tuple = vae_loss(date_val, recon_v, mean_v, logvar_v)
            v_loss = v_loss_tuple[0]
            val_loss_total = val_loss_total + float(v_loss)
            n_val_batches = n_val_batches + 1

        if n_val_batches > 0:
            val_loss_total = val_loss_total / n_val_batches
        else:
            val_loss_total = 0.0

        train_losses.append(epoch_loss)
        val_losses.append(val_loss_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  Train: {epoch_loss:.4f}  Val: {val_loss_total:.4f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    enc_path = os.path.join(WEIGHTS_DIR, "encoder.weights.h5")
    dec_path = os.path.join(WEIGHTS_DIR, "decoder.weights.h5")
    
    encoder.save_weights(enc_path)
    decoder.save_weights(dec_path)
    
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="VAE",
        save_path="plots/vae_loss.png",
    )

    return decoder


if __name__ == "__main__":
    train()

