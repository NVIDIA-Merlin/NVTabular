import copy

import tensorflow as tf

from nvtabular.feature_group import FeatureGroup


class FilterFeatures(tf.keras.layers.Layer):
    def __init__(self, columns, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.columns = columns

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.columns}

        return outputs

    def get_config(self):
        return {
            "columns": self.columns,
        }


class ConcatFeatures(tf.keras.layers.Layer):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        features = inputs
        if isinstance(inputs, dict):
            features = [v for k, v in sorted(inputs.items())]

        return tf.concat(features, axis=self.axis)

    def get_config(self):
        return {
            "axis": self.axis,
        }


class StackFeatures(tf.keras.layers.Layer):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        features = [v for k, v in sorted(inputs.items())]

        return tf.stack(features, axis=self.axis)

    def get_config(self):
        return {
            "axis": self.axis,
        }


class TabularLayer(tf.keras.layers.Layer):
    def __call__(self, inputs, pre=None, post=None, merge_with=None, stack_outputs=False, concat_outputs=False,
                 filter_columns=None,
                 **kwargs):
        if concat_outputs:
            post = ConcatFeatures()
        if stack_outputs:
            post = StackFeatures()
        if filter_columns:
            pre = FilterFeatures(filter_columns)
        if pre:
            inputs = pre(inputs)
        outputs = super().__call__(inputs, **kwargs)

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer in merge_with:
                outputs.update(layer(inputs))

        if post:
            outputs = post(outputs)

        return outputs

    def call_on_cols(self, inputs, columns_to_include):
        return self(inputs, pre=FilterFeatures(columns_to_include))

    def call_and_concat(self, inputs):
        return self(inputs, post=ConcatFeatures())

    def call_and_stack(self, inputs):
        return self(inputs, post=StackFeatures())

    def apply_to_all(self, inputs, columns_to_filter=None):
        outputs = {}
        if columns_to_filter:
            inputs = FilterFeatures(columns_to_filter)(inputs)
        for key, val in inputs.items():
            outputs[key] = self(val)

        return outputs


class SequentialLayer(TabularLayer):
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

        super(SequentialLayer, self).__init__(**kwargs)
        self.filter_features = filter_features
        if filter_features:
            self.layers = [FilterFeatures(filter_features), *copy.copy(layers)]
        else:
            self.layers = copy.copy(layers)

    def compute_output_shape(self, input_shape):
        output_shape = tf.TensorShape(input_shape)
        for l in self.layers:
            output_shape = l.compute_output_shape(output_shape)
        return tf.TensorShape(output_shape)

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

    def call(self, inputs, training=False):
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


def right_shift(self, other):
    if isinstance(other, list):
        left_side = [FilterFeatures(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialLayer) else [other]
    right_side = self.layers if isinstance(self, SequentialLayer) else [self]

    return SequentialLayer(left_side + right_side)


tf.keras.layers.Layer.__rrshift__ = right_shift


# class TFFeatureGroup(FeatureGroup):
#     def __call__(self, operator, **kwargs):
#         pass
