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

COND_DIM   = 22
DATE_DIM   = 3
LATENT_DIM = 16


def build_encoder():
    encoder_layers = [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(COND_DIM + DATE_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(LATENT_DIM * 2)
    ]
    model = tf.keras.Sequential(encoder_layers, name="Encoder")
    return model


def build_decoder():
    decoder_layers = [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(LATENT_DIM + COND_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(DATE_DIM, activation="sigmoid")
    ]
    model = tf.keras.Sequential(decoder_layers, name="Decoder")
    return model


def reparameterise(mean, log_var):
    shape_of_mean = tf.shape(mean)
    eps = tf.random.normal(shape=shape_of_mean)
    
    std_dev = 0.5 * log_var
    std = tf.exp(std_dev)
    
    random_z = eps * std
    z = mean + random_z
    return z

