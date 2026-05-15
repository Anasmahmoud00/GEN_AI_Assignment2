"""
model/flow/train.py
-------------------
Training loop for the Normalizing Flow (RealNVP).

Run:
    python model/flow/train.py

Saves weights to: model/weights/flow/

Flow Loss = Negative Log-Likelihood
------------------------------------------
We want to maximise the probability of the real dates under the model.

log p(x|c) = log p(z) + log|det(Jacobian)|

Where:
  - z = f(x|c)                     ← forward pass through flow
  - log p(z) = Gaussian log-prob   ← z should look like N(0,1)
  - log|det(Jacobian)|             ← accounts for volume change

We MINIMISE the negative of this (standard practice).
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import ConditionalRealNVP

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42
DATA_PATH   = "data/data.txt" if os.path.exists("data/data.txt") else "data.txt"
WEIGHTS_DIR = "model/weights/flow"


# ── Log-likelihood of Gaussian ────────────────────────────────────────────────

def gaussian_log_prob(z):
    """
    Log probability of z under standard Normal N(0, I).

    log p(z) = -0.5 * sum(z^2 + log(2π))
    """
    log_2pi = tf.math.log(2.0 * 3.14159265)
    return -0.5 * tf.reduce_sum(tf.square(z) + log_2pi, axis=1)


# ── Flow loss = negative log-likelihood ───────────────────────────────────────

def flow_loss(x, condition, flow):
    """
    Compute the negative log-likelihood for one batch.

    = -mean( log p(z) + log_det )
    """
    z, log_det = flow.forward(x, condition)
    log_prob   = gaussian_log_prob(z) + log_det
    return -tf.reduce_mean(log_prob)


# ── One training step ─────────────────────────────────────────────────────────

@tf.function
def train_step(conditions, real_dates, flow, optimizer):
    """
    One batch update using tf.GradientTape (NO model.fit).
    """
    with tf.GradientTape() as tape:
        loss = flow_loss(real_dates, conditions, flow)

    grads = tape.gradient(loss, flow.trainable_variables)
    optimizer.apply_gradients(zip(grads, flow.trainable_variables))
    return loss


# ── Main training function ────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(DATA_PATH)
    print(f"Training Normalizing Flow on {len(X_train)} samples …\n")

    flow      = ConditionalRealNVP()
    optimizer = tf.keras.optimizers.Adam(LR)

    # build by running a dummy forward pass
    dummy_x = tf.zeros((1, 3))
    dummy_c = tf.zeros((1, 22))
    flow.forward(dummy_x, dummy_c)
    print(f"Flow built — {len(flow.trainable_variables)} variable tensors\n")

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            cond_batch = tf.constant(X_train[i : i + BATCH_SIZE])
            date_batch = tf.constant(Y_train[i : i + BATCH_SIZE])

            loss       = train_step(cond_batch, date_batch, flow, optimizer)
            epoch_loss += float(loss)
            n_batches  += 1

        epoch_loss /= n_batches

        # validation loss
        val_loss = float(flow_loss(
            tf.constant(Y_val), tf.constant(X_val), flow
        ))

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"Train NLL: {epoch_loss:.4f}  Val NLL: {val_loss:.4f}")

    # save weights
    flow.save_weights(WEIGHTS_DIR)

    # save loss plot
    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="Normalizing Flow (RealNVP)",
        save_path="plots/flow_loss.png",
    )

    return flow


if __name__ == "__main__":
    train()
