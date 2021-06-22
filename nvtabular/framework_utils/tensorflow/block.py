import copy
from typing import Union

import tensorflow as tf

from nvtabular.framework_utils.tensorflow import features
from nvtabular.framework_utils.tensorflow.heads import Head
from nvtabular.framework_utils.tensorflow.tfrs import Model


class Block(features.TabularLayer):
    """The SequentialLayer represents a sequence of Keras layers.
    It is a Keras Layer that can be used instead of tf.keras.layers.Sequential,
    which is actually a Keras Model.  In contrast to keras Sequential, this
    layer can be used as a pure Layer in tf.functions and when exporting
    SavedModels, without having to pre-declare input and output shapes.  In turn,
    this layer is usable as a preprocessing layer for TF Agents Networks, and
    can be exported via PolicySaver.
    Usage:
    ```python
    c = SequentialLayer([layer1, layer2, layer3])
    output = c(inputs)    # Equivalent to: output = layer3(layer2(layer1(inputs)))
    ```
    """

    def __init__(self, layers, filter_features=None, **kwargs):
        """Create a composition.
        Args:
          layers: A list or tuple of layers to compose.
          **kwargs: Arguments to pass to `Keras` layer initializer, including
            `name`.
        Raises:
          TypeError: If any of the layers are not instances of keras `Layer`.
        """
        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'"
                        .format(layer))

        super(Block, self).__init__(**kwargs)
        self.filter_features = filter_features
        if filter_features:
            self.layers = [features.FilterFeatures(filter_features), *copy.copy(layers)]
        else:
            self.layers = copy.copy(layers)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for l in self.layers:
            output_shape = l.compute_output_shape(output_shape)
        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = input_signature
        for l in self.layers:
            output_signature = l.compute_output_signature(output_signature)
        return output_signature

    def build(self, input_shape=None):
        for l in self.layers:
            l.build(input_shape)
            input_shape = l.compute_output_shape(input_shape)
        self.built = True

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = {}
        for l in self.layers:
            for v in l.trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def non_trainable_weights(self):
        weights = {}
        for l in self.layers:
            for v in l.non_trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def trainable(self):
        return all([l.trainable for l in self.layers])

    @trainable.setter
    def trainable(self, value):
        for l in self.layers:
            l.trainable = value

    @property
    def losses(self):
        values = set()
        for l in self.layers:
            values.update(l.losses)
        return list(values)

    @property
    def regularizers(self):
        values = set()
        for l in self.layers:
            values.update(l.regularizers)
        return list(values)

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        for l in self.layers:
            outputs = l(outputs, training=training)
        return outputs

    def get_config(self):
        config = {"filter_features": self.filter_features}
        for i, layer in enumerate(self.layers):
            config[i] = {
                'class_name': layer.__class__.__name__,
                'config': copy.deepcopy(layer.get_config())
            }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]
        return cls(layers)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def __rshift__(self, other):
        return right_shift_layer(other, self)

    def with_head(self, head: Head, **kwargs):
        return BlockWithHead(self, head, **kwargs)


BlockType = Union[tf.keras.layers.Layer, Block]


class BlockWithHead(Model):
    def __init__(self, block: BlockType, head: Head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        self.head = head

    def call(self, inputs, **kwargs):
        return self.head(self.block(inputs, **kwargs), **kwargs)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        targets = self.head.pop_labels(inputs)
        logits = self(inputs, training=training)

        return self.head.compute_loss(targets, logits)

    def get_config(self):
        pass


def right_shift_layer(self, other):
    if isinstance(other, list):
        left_side = [features.FilterFeatures(other)]
    else:
        left_side = other.layers if isinstance(other, Block) else [other]
    right_side = self.layers if isinstance(self, Block) else [self]

    return Block(left_side + right_side)