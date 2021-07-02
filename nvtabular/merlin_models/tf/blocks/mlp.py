from merlin_models.tf.blocks.base import SequentialBlock

import tensorflow as tf


class MLPBlock(SequentialBlock):
    def __init__(self, dimensions, activation="relu", use_bias: bool = True, filter_features=None, **kwargs):
        layers = [tf.keras.layers.Dense(
            dim,
            activation=activation,
            use_bias=use_bias) for dim in dimensions]

        super().__init__(layers, filter_features, **kwargs)
