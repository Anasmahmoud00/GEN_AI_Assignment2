"""
model/flow/model.py
-------------------
Conditional Normalizing Flow (RealNVP) for the Dates Generator.

What is a Normalizing Flow?
---------------------------
  - the data space  x  (the date vector, 3 dims)
  - a simple space  z  (Gaussian noise, 3 dims)

Such that:   z = f(x | condition)     ← forward  (training)
             x = f⁻¹(z | condition)  ← inverse  (inference)
"""

import os
import tensorflow as tf

COND_DIM = 22
DATE_DIM = 3
HALF1    = 1
HALF2    = 2


def build_scale_translate_net(input_dim, cond_dim, name):
    input_shape = (input_dim + cond_dim,)
    inp = tf.keras.Input(shape=input_shape)
    
    dense1 = tf.keras.layers.Dense(32, activation="relu")
    x1 = dense1(inp)
    
    dense2 = tf.keras.layers.Dense(32, activation="relu")
    x2 = dense2(x1)
    
    output_dim = DATE_DIM - input_dim
    
    dense_s = tf.keras.layers.Dense(output_dim)
    s = dense_s(x2)
    
    dense_t = tf.keras.layers.Dense(output_dim)
    t = dense_t(x2)
    
    model = tf.keras.Model(inp, [s, t], name=name)
    return model


class CouplingLayer(tf.keras.layers.Layer):
    def __init__(self, mask, name):
        super().__init__(name=name)
        self.mask = mask

        if mask == 0:
            fixed_dim       = HALF1
            transformed_dim = HALF2
        else:
            fixed_dim       = HALF2
            transformed_dim = HALF1

        st_name = name + "_st"
        self.st_net = build_scale_translate_net(
            input_dim=fixed_dim,
            cond_dim=COND_DIM,
            name=st_name
        )

    def forward(self, x, condition):
        if self.mask == 0:
            x1 = x[:, :1]
            x2 = x[:, 1:]
        else:
            x2 = x[:, :1]
            x1 = x[:, 1:]

        st_input = tf.concat([x1, condition], axis=1)
        s_raw, t = self.st_net(st_input)
        s = tf.tanh(s_raw)

        exp_s = tf.exp(s)
        scaled_x2 = x2 * exp_s
        z2 = scaled_x2 + t
        
        log_det = tf.reduce_sum(s, axis=1)

        if self.mask == 0:
            z = tf.concat([x1, z2], axis=1)
        else:
            z = tf.concat([z2, x1], axis=1)

        return z, log_det

    def inverse(self, z, condition):
        if self.mask == 0:
            z1 = z[:, :1]
            z2 = z[:, 1:]
        else:
            z2 = z[:, :1]
            z1 = z[:, 1:]

        st_input = tf.concat([z1, condition], axis=1)
        s_raw, t = self.st_net(st_input)
        s = tf.tanh(s_raw)

        diff = z2 - t
        neg_s = -s
        exp_neg_s = tf.exp(neg_s)
        x2 = diff * exp_neg_s

        if self.mask == 0:
            x = tf.concat([z1, x2], axis=1)
        else:
            x = tf.concat([x2, z1], axis=1)

        return x


class ConditionalRealNVP:
    def __init__(self):
        layer0 = CouplingLayer(mask=0, name="coupling_0")
        layer1 = CouplingLayer(mask=1, name="coupling_1")
        layer2 = CouplingLayer(mask=0, name="coupling_2")
        layer3 = CouplingLayer(mask=1, name="coupling_3")
        
        self.layers_list = [layer0, layer1, layer2, layer3]

    @property
    def trainable_variables(self):
        all_vars = []
        for layer in self.layers_list:
            vars_in_layer = layer.st_net.trainable_variables
            for v in vars_in_layer:
                all_vars.append(v)
        return all_vars

    def forward(self, x, condition):
        z = x
        batch_size = tf.shape(x)[0]
        total_log_det = tf.zeros(batch_size)

        for layer in self.layers_list:
            z, log_det = layer.forward(z, condition)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z, condition):
        x = z
        reversed_layers = []
        for layer in self.layers_list:
            reversed_layers.insert(0, layer)
            
        for layer in reversed_layers:
            x = layer.inverse(x, condition)
        return x

    def save_weights(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for layer in self.layers_list:
            file_name = layer.name + ".weights.h5"
            path = os.path.join(directory, file_name)
            layer.st_net.save_weights(path)
        print(f"Flow weights saved to {directory}/")

    def load_weights(self, directory):
        dummy_x = tf.zeros((1, DATE_DIM))
        dummy_c = tf.zeros((1, COND_DIM))
        self.forward(dummy_x, dummy_c)

        for layer in self.layers_list:
            file_name = layer.name + ".weights.h5"
            path = os.path.join(directory, file_name)
            layer.st_net.load_weights(path)
        print(f"Flow weights loaded from {directory}/")
