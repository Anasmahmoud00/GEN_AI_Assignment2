"""
model/vae/model.py
------------------
Conditional VAE (CVAE) for the Dates Generator.

Architecture
------------
Encoder
  Input  : condition vector (22) + real date (3)  →  concat = 25
  Layers : Dense(64) → Dense(32)
  Output : mean (16) and log_variance (16)  ← latent space

Decoder  (also called the Generator at inference)
  Input  : latent z (16) + condition vector (22)  →  concat = 38
  Layers : Dense(64) → Dense(3)
  Output : reconstructed date vector (3 floats, values in [0,1])

At inference we do NOT use the encoder.
We sample z ~ N(0,1) and decode with the condition.
"""

import tensorflow as tf

COND_DIM   = 22   # condition vector size
DATE_DIM   = 3    # date vector size (day, month, year normalised)
LATENT_DIM = 16   # size of the latent space z


def build_encoder():
    """
    Encoder E(c, x) → (mean, log_var)

    Takes condition + real date, outputs latent distribution parameters.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",
                              input_shape=(COND_DIM + DATE_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(LATENT_DIM * 2),   # first half = mean, second = log_var
    ], name="Encoder")
    return model


def build_decoder():
    """
    Decoder D(z, c) → reconstructed date

    Takes latent z + condition, outputs a date vector in (0,1).
    This is also used standalone at inference time.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",
                              input_shape=(LATENT_DIM + COND_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(DATE_DIM, activation="sigmoid"),  # output in (0,1)
    ], name="Decoder")
    return model


def reparameterise(mean, log_var):
    """
    Reparameterisation trick:  z = mean + eps * std
    Allows gradients to flow through the sampling step.
    """
    eps = tf.random.normal(shape=tf.shape(mean))
    std = tf.exp(0.5 * log_var)
    return mean + eps * std
