"""
model/flow/train.py
-------------------
Training loop for the Normalizing Flow (RealNVP).

Run:
    python model/flow/train.py

Saves weights to: model/weights/flow/

Flow Loss = Negative Log-Likelihood
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
from model import ConditionalRealNVP

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42

if os.path.exists("data/data.txt"):
    DATA_PATH = "data/data.txt"
else:
    DATA_PATH = "data.txt"

WEIGHTS_DIR = "model/weights/flow"


def gaussian_log_prob(z):
    pi_val = 3.14159265
    two_pi = 2.0 * pi_val
    log_2pi = tf.math.log(two_pi)
    
    z_sq = tf.square(z)
    sum_val = z_sq + log_2pi
    
    reduced_sum = tf.reduce_sum(sum_val, axis=1)
    result = -0.5 * reduced_sum
    return result


def flow_loss(x, condition, flow):
    z, log_det = flow.forward(x, condition)
    
    log_p_z = gaussian_log_prob(z)
    total_log_prob = log_p_z + log_det
    
    mean_log_prob = tf.reduce_mean(total_log_prob)
    neg_nll = -mean_log_prob
    return neg_nll


@tf.function
def train_step(conditions, real_dates, flow, optimizer):
    with tf.GradientTape() as tape:
        loss = flow_loss(real_dates, conditions, flow)

    train_vars = flow.trainable_variables
    grads = tape.gradient(loss, train_vars)
    
    grads_and_vars = zip(grads, train_vars)
    optimizer.apply_gradients(grads_and_vars)
    
    return loss


def train():
    set_seed(SEED)

    data_all = load_data(DATA_PATH)
    X_train = data_all[0]
    Y_train = data_all[1]
    X_val   = data_all[2]
    Y_val   = data_all[3]
    X_test  = data_all[4]
    Y_test  = data_all[5]
    
    print(f"Training Normalizing Flow on {len(X_train)} samples ...\n")

    flow      = ConditionalRealNVP()
    optimizer = tf.keras.optimizers.Adam(LR)

    dummy_x = tf.zeros((1, 3))
    dummy_c = tf.zeros((1, 22))
    flow.forward(dummy_x, dummy_c)
    
    train_vars_count = len(flow.trainable_variables)
    print(f"Flow built - {train_vars_count} variable tensors\n")

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            end_idx = i + BATCH_SIZE
            cond_batch = tf.constant(X_train[i : end_idx])
            date_batch = tf.constant(Y_train[i : end_idx])

            loss_val   = train_step(cond_batch, date_batch, flow, optimizer)
            epoch_loss = epoch_loss + float(loss_val)
            n_batches  = n_batches + 1

        epoch_loss = epoch_loss / n_batches

        Y_val_tf = tf.constant(Y_val)
        X_val_tf = tf.constant(X_val)
        val_loss_val = flow_loss(Y_val_tf, X_val_tf, flow)
        val_loss = float(val_loss_val)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  Train NLL: {epoch_loss:.4f}  Val NLL: {val_loss:.4f}")

    flow.save_weights(WEIGHTS_DIR)

    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="Normalizing Flow (RealNVP)",
        save_path="plots/flow_loss.png",
    )

    return flow


if __name__ == "__main__":
    train()

