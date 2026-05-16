"""
model/diffusion/model.py
------------------------
Simplified Conditional Diffusion Model (Denoising MLP).

This model learns to predict the noise added to a date vector.
Given (noisy_date, condition, time_step), it outputs the predicted noise.
"""

import tensorflow as tf

COND_DIM = 22
DATE_DIM = 3


def build_diffusion_model():
    # Inputs
    noisy_date_input = tf.keras.Input(shape=(DATE_DIM,), name="noisy_date")
    condition_input  = tf.keras.Input(shape=(COND_DIM,), name="condition")
    time_input       = tf.keras.Input(shape=(1,),        name="time_step")

    # Combine all inputs into one vector
    # We concatenate: [noisy_date, condition, time_step]
    x = tf.keras.layers.Concatenate()([noisy_date_input, condition_input, time_input])

    # Simple Dense layers (MLP)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Output: Predicted noise (same shape as date vector)
    output = tf.keras.layers.Dense(DATE_DIM)(x)

    model = tf.keras.Model(
        inputs=[noisy_date_input, condition_input, time_input],
        outputs=output,
        name="Diffusion_Denoiser"
    )
    return model
