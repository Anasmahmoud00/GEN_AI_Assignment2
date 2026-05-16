"""
model/gan/model.py
------------------
Conditional GAN for the Dates Generator.

Architecture
------------
Generator
  Input  : noise vector (64) + condition vector (22)  →  concat = 86
  Layers : Dense(128) → Dense(64) → Dense(3)
  Output : fake date vector (3 floats, values in [0,1])

Discriminator
  Input  : condition vector (22) + date vector (3)    →  concat = 25
  Layers : Dense(64) → Dense(32) → Dense(1)
  Output : single float (probability = real)
"""

import tensorflow as tf

NOISE_DIM    = 64
COND_DIM     = 22
DATE_DIM     = 3


def build_generator():
    gen_layers = [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(NOISE_DIM + COND_DIM,)),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(DATE_DIM, activation="sigmoid")
    ]
    model = tf.keras.Sequential(gen_layers, name="Generator")
    return model


def build_discriminator():
    disc_layers = [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(COND_DIM + DATE_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1,  activation="sigmoid")
    ]
    model = tf.keras.Sequential(disc_layers, name="Discriminator")
    return model

