"""
model/mlp/model.py
------------------
MLP (Multi-Layer Perceptron) for the Dates Generator.

This is a simple supervised generative model:
  - learns a direct mapping from conditions → date
  - NOT a probabilistic model (no noise/sampling)
  - at inference: pass condition → get predicted date directly

Architecture
------------
  Input  : condition vector (22)
  Layers : Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(3)
  Output : date vector (3 floats in [0,1])

Why outside the course?
  The course focuses on generative models (GAN, VAE, Diffusion).
  MLP as a conditional regressor / conditional generator is not
  part of the generative models curriculum — it's a baseline approach.
"""

import tensorflow as tf

COND_DIM = 22
DATE_DIM = 3


def build_mlp():
    """
    Simple MLP: conditions (22) → date (3).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu",
                              input_shape=(COND_DIM,)),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(32,  activation="relu"),
        tf.keras.layers.Dense(DATE_DIM, activation="sigmoid"),  # output in (0,1)
    ], name="MLP")
    return model
