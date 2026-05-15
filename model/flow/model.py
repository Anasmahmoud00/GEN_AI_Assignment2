"""
model/flow/model.py
-------------------
Conditional Normalizing Flow (RealNVP) for the Dates Generator.

What is a Normalizing Flow?
---------------------------
A normalizing flow learns an **invertible** transformation f between:
  - the data space  x  (the date vector, 3 dims)
  - a simple space  z  (Gaussian noise, 3 dims)

Such that:   z = f(x | condition)     ← forward  (training)
             x = f⁻¹(z | condition)  ← inverse  (inference)

At inference we:
  1. Sample z ~ N(0, 1)
  2. Run the inverse transformation x = f⁻¹(z | condition)
  3. x is our generated date

Why RealNVP?
  RealNVP (Real-Valued Non-Volume Preserving) uses "coupling layers"
  which are easy to invert analytically — no need to solve equations.

Coupling Layer Logic
--------------------
  Split x into two halves: x1 (dim 1) and x2 (dim 2)

  Forward (x → z):
    z1 = x1                          ← unchanged
    z2 = x2 * exp(s(x1,c)) + t(x1,c) ← scaled + shifted

  Inverse (z → x):
    x1 = z1
    x2 = (z2 - t(z1,c)) * exp(-s(z1,c))

  Where s() and t() are small neural networks that also take
  the condition vector c as input.

We stack 4 coupling layers, alternating which half is "active".
"""

import os
import tensorflow as tf

COND_DIM = 22
DATE_DIM = 3      # 3-dim date vector: (day, month, year) normalised
HALF1    = 1      # first split:  x[:, :1]
HALF2    = 2      # second split: x[:, 1:]


def build_scale_translate_net(input_dim, cond_dim, name):
    """
    Small network that outputs scale (s) and translate (t) vectors.
    Used inside each coupling layer.

    Input  : (half_of_date + condition)
    Output : scale and translate for the other half
    """
    inp    = tf.keras.Input(shape=(input_dim + cond_dim,))
    x      = tf.keras.layers.Dense(32, activation="relu")(inp)
    x      = tf.keras.layers.Dense(32, activation="relu")(x)
    s      = tf.keras.layers.Dense(DATE_DIM - input_dim)(x)   # scale
    t      = tf.keras.layers.Dense(DATE_DIM - input_dim)(x)   # translate
    return tf.keras.Model(inp, [s, t], name=name)


class CouplingLayer(tf.keras.layers.Layer):
    """
    One RealNVP coupling layer.

    mask=0  →  x1 = x[:, :1],  x2 = x[:, 1:]  (x1 is fixed, x2 is transformed)
    mask=1  →  x1 = x[:, 1:],  x2 = x[:, :1]  (x1 is fixed, x2 is transformed)
    """

    def __init__(self, mask, name):
        super().__init__(name=name)
        self.mask = mask

        if mask == 0:
            fixed_dim       = HALF1   # 1 dim fixed
            transformed_dim = HALF2   # 2 dims transformed
        else:
            fixed_dim       = HALF2
            transformed_dim = HALF1

        self.st_net = build_scale_translate_net(
            input_dim=fixed_dim,
            cond_dim=COND_DIM,
            name=f"{name}_st"
        )

    def forward(self, x, condition):
        """x → z  (used during training)"""
        if self.mask == 0:
            x1, x2 = x[:, :1], x[:, 1:]
        else:
            x2, x1 = x[:, :1], x[:, 1:]

        st_input = tf.concat([x1, condition], axis=1)
        s, t     = self.st_net(st_input)
        s        = tf.tanh(s)                      # keep s bounded

        z2       = x2 * tf.exp(s) + t
        log_det  = tf.reduce_sum(s, axis=1)        # log|det(Jacobian)|

        if self.mask == 0:
            z = tf.concat([x1, z2], axis=1)
        else:
            z = tf.concat([z2, x1], axis=1)

        return z, log_det

    def inverse(self, z, condition):
        """z → x  (used during inference)"""
        if self.mask == 0:
            z1, z2 = z[:, :1], z[:, 1:]
        else:
            z2, z1 = z[:, :1], z[:, 1:]

        st_input = tf.concat([z1, condition], axis=1)
        s, t     = self.st_net(st_input)
        s        = tf.tanh(s)

        x2 = (z2 - t) * tf.exp(-s)

        if self.mask == 0:
            x = tf.concat([z1, x2], axis=1)
        else:
            x = tf.concat([x2, z1], axis=1)

        return x


class ConditionalRealNVP:
    """
    Stack of 4 coupling layers alternating mask=0 and mask=1.
    """

    def __init__(self):
        self.layers_ = [
            CouplingLayer(mask=0, name="coupling_0"),
            CouplingLayer(mask=1, name="coupling_1"),
            CouplingLayer(mask=0, name="coupling_2"),
            CouplingLayer(mask=1, name="coupling_3"),
        ]

    @property
    def trainable_variables(self):
        """Collect all trainable variables from all coupling layers."""
        vars_ = []
        for layer in self.layers_:
            vars_ += layer.st_net.trainable_variables
        return vars_

    def forward(self, x, condition):
        """
        Run all coupling layers forward: x → z
        Returns (z, total_log_det)
        """
        z           = x
        total_log_det = tf.zeros(tf.shape(x)[0])

        for layer in self.layers_:
            z, log_det    = layer.forward(z, condition)
            total_log_det += log_det

        return z, total_log_det

    def inverse(self, z, condition):
        """
        Run all coupling layers in reverse: z → x
        Used at inference to generate dates.
        """
        x = z
        for layer in reversed(self.layers_):
            x = layer.inverse(x, condition)
        return x

    def save_weights(self, directory):
        """Save each coupling layer's network weights."""
        import os
        os.makedirs(directory, exist_ok=True)
        for layer in self.layers_:
            path = os.path.join(directory, f"{layer.name}.weights.h5")
            layer.st_net.save_weights(path)
        print(f"Flow weights saved to {directory}/")

    def load_weights(self, directory):
        """Load weights back into each coupling layer."""
        # Build the networks first with a dummy forward pass
        dummy_x = tf.zeros((1, DATE_DIM))
        dummy_c = tf.zeros((1, COND_DIM))
        self.forward(dummy_x, dummy_c)

        for layer in self.layers_:
            path = os.path.join(directory, f"{layer.name}.weights.h5")
            layer.st_net.load_weights(path)
        print(f"Flow weights loaded from {directory}/")
