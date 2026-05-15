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

Both are plain tf.keras.Sequential — no subclassing complexity.
"""

import tensorflow as tf

NOISE_DIM    = 64   # size of the random noise fed to the generator
COND_DIM     = 22   # condition vector size  (from tokenizer)
DATE_DIM     = 3    # output date vector size (day, month, year normalised)


def build_generator():
    """
    Generator G(z, c) → fake_date

    Takes noise z and condition c, outputs a date vector in (0, 1).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu",
                              input_shape=(NOISE_DIM + COND_DIM,)),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(DATE_DIM, activation="sigmoid"),  # output in (0,1)
    ], name="Generator")
    return model


def build_discriminator():
    """
    Discriminator D(c, date) → probability(real)

    Takes condition + date (real or fake), outputs a single score.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",
                              input_shape=(COND_DIM + DATE_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1,  activation="sigmoid"),  # 1 = real, 0 = fake
    ], name="Discriminator")
    return model
