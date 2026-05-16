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

root_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(root_path)

from src.dataset  import load_data
from src.utils    import set_seed
from src.evaluate import plot_loss
from model import build_mlp

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.001
SEED        = 42

if os.path.exists("data/data.txt"):
    DATA_PATH = "data/data.txt"
else:
    DATA_PATH = "data.txt"

WEIGHTS_DIR = "model/weights/mlp"

mse_loss_obj = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(conditions, real_dates, model, optimizer):
    with tf.GradientTape() as tape:
        predicted = model(conditions, training=True)
        loss      = mse_loss_obj(real_dates, predicted)

    train_vars = model.trainable_variables
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
    
    print(f"Training MLP on {len(X_train)} samples ...\n")

    model     = build_mlp()
    model.summary()
    
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

            loss_val   = train_step(cond_batch, date_batch, model, optimizer)
            epoch_loss = epoch_loss + float(loss_val)
            n_batches  = n_batches + 1

        epoch_loss = epoch_loss / n_batches

        X_val_tf = tf.constant(X_val)
        val_preds = model(X_val_tf, training=False)
        
        val_preds_np = val_preds.numpy()
        val_loss_val = mse_loss_obj(Y_val, val_preds_np)
        val_loss = float(val_loss_val)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS}  Train MSE: {epoch_loss:.4f}  Val MSE: {val_loss:.4f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    save_path = os.path.join(WEIGHTS_DIR, "mlp.weights.h5")
    model.save_weights(save_path)
    
    print(f"\nWeights saved to {WEIGHTS_DIR}/")

    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        model_name="MLP",
        save_path="plots/mlp_loss.png",
    )

    return model


if __name__ == "__main__":
    train()

