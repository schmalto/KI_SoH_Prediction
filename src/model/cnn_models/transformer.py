# Write a transformer model with the following architecture:
#
# The model should have an embedding layer, a positional encoding layer, and two transformer blocks.
# The embedding layer should use 128 units.
# The positional encoding layer should use 128 units.
# The transformer blocks should use 128 units and 2 heads.
# The model should output a single value.
# The model should use the mean squared error loss function.
# The model should use the Adam optimizer with a learning rate of 0.001.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import tf.keras as keras
from keras.layers import MultiHeadAttention, Dense, Embedding, Dropout
from model_train_helpers import PlotLossAccuracy, generate_training_data

class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)
    
class Transformer
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = layers.Embedding(128)
        self.positional_encoding = PositionalEncoding()
        self.transformer_block1 = layers.MultiHeadAttention(num_heads=2, key_dim=128)
        self.transformer_block2 = layers.MultiHeadAttention(num_heads=2, key_dim=128)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1)
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_block1(x, x)
        x = self.transformer_block2(x, x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def compile_model(self):
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return self
    
def main():
    model = Transformer()
    model.compile_model()
    model.summary()
    dataset = generate_training_data()
    n_epochs = 500
    pltCallBack = PlotLossAccuracy("transformer")
    history = model.fit(dataset, epochs=n_epochs, batch_size=32, callbacks=[pltCallBack])
    return model, history
