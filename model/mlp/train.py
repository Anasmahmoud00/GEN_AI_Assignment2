"""
model/mlp/train.py
------------------
Training loop for the MLP model.

Run:
    python model/mlp/train.py

Saves weights to:  model/weights/mlp/mlp.weights.h5

Loss: MSE (Mean Squared Error)
  We minimise the distance between the predicted date vector
  and the real date vector.
"""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_mlp

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42
DATA_PATH   = "data/data.txt" if os.path.exists("data/data.txt") else "data.txt"
WEIGHTS_DIR = "model/weights/mlp"

mse = tf.keras.losses.MeanSquaredError()


# ── One training step ─────────────────────────────────────────────────────────

@tf.function
def train_step(conditions, real_dates, model, optimizer):
    """
    One batch update using tf.GradientTape (NO model.fit).

    Steps
    -----
    1. Forward pass  →  predicted date
    2. Compute MSE loss
    3. Backprop and update weights
    """
    with tf.GradientTape() as tape:
        predicted = model(conditions, training=True)
        loss      = mse(real_dates, predicted)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ── Main training function ────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(DATA_PATH)
    print(f"Training MLP on {len(X_train)} samples …\n")

    model     = build_mlp()
    model.summary()
    optimizer = tf.keras.optimizers.Adam(LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            cond_batch = tf.constant(X_train[i : i + BATCH_SIZE])
            date_batch = tf.constant(Y_train[i : i + BATCH_SIZE])

            loss       = train_step(cond_batch, date_batch, model, optimizer)
            epoch_loss += float(loss)
            n_batches  += 1

        epoch_loss /= n_batches

        # validation loss (no gradient)
        val_preds = model(tf.constant(X_val), training=False)
        val_loss  = float(mse(Y_val, val_preds.numpy()))

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"Train MSE: {epoch_loss:.4f}  Val MSE: {val_loss:.4f}")

    # save weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save_weights(os.path.join(WEIGHTS_DIR, "mlp.weights.h5"))
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    # save loss plot
    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="MLP",
        save_path="plots/mlp_loss.png",
    )

    return model


if __name__ == "__main__":
    train()
