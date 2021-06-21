import copy

import tensorflow as tf

from nvtabular.column_group import ColumnGroup
from nvtabular.feature_group import FeatureGroup


class FilterFeatures(tf.keras.layers.Layer):
    def __init__(self, columns, trainable=False, name=None, dtype=None, dynamic=False, pop=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.columns = columns
        self.pop = pop

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, dict), "Inputs needs to be a dict"

        outputs = {k: v for k, v in inputs.items() if k in self.columns}
        if self.pop:
            for key in outputs.keys():
                inputs.pop(key)

        return outputs

    def get_config(self):
        return {
            "columns": self.columns,
        }


class ConcatFeatures(tf.keras.layers.Layer):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        return tf.concat(tf.nest.flatten(tf.nest.map_structure(self.flatten, inputs)), axis=self.axis)

    def get_config(self):
        return {
            "axis": self.axis,
        }


class AsTabular(tf.keras.layers.Layer):
    def __init__(self, output_name, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.output_name = output_name

    def call(self, inputs, **kwargs):
        return {self.output_name: inputs}

    def get_config(self):
        return {
            "axis": self.axis,
        }


class StackFeatures(tf.keras.layers.Layer):
    def __init__(self, axis=-1, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        return tf.stack(tf.nest.flatten(tf.nest.map_structure(self.flatten, inputs)), axis=self.axis)

    def get_config(self):
        return {
            "axis": self.axis,
        }


class TabularLayer(tf.keras.layers.Layer):
    def __init__(self, aggregation=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.aggregation = aggregation

    def __call__(self, inputs, pre=None, post=None, merge_with=None, stack_outputs=False, concat_outputs=False,
                 filter_columns=None, **kwargs):
        post_op = self.maybe_aggregate()
        if concat_outputs:
            post_op = ConcatFeatures()
        if stack_outputs:
            post_op = StackFeatures()
        if filter_columns:
            pre = FilterFeatures(filter_columns)
        if pre:
            inputs = pre(inputs)
        outputs = super().__call__(inputs, **kwargs)

        if merge_with:
            if not isinstance(merge_with, list):
                merge_with = [merge_with]
            for layer_or_tensor in merge_with:
                to_add = layer_or_tensor(inputs) if callable(layer_or_tensor) else layer_or_tensor
                outputs.update(to_add)

        if post_op:
            outputs = post_op(outputs)

        return outputs

    def maybe_aggregate(self):
        if self.aggregation == "concat":
            return ConcatFeatures()

        if self.aggregation == "stack":
            return StackFeatures()

        return None

    def compute_output_shape(self, input_shapes):
        batch_size = [i for i in input_shapes.values()][0][0]
        if self.aggregation == "concat":
            return batch_size, sum([i[1] for i in input_shapes.values()])
        elif self.aggregation == "stack":
            last_dim = [i for i in input_shapes.values()][0][-1]

            return batch_size, len(input_shapes), last_dim
        else:
            return super(TabularLayer, self).compute_output_shape(input_shapes)

    def call_on_cols(self, inputs, columns_to_include):
        return self(inputs, pre=FilterFeatures(columns_to_include))

    def call_and_concat(self, inputs):
        return self(inputs, post=ConcatFeatures())

    def call_and_stack(self, inputs):
        return self(inputs, post=StackFeatures())

    def apply_to_all(self, inputs, columns_to_filter=None):
        if columns_to_filter:
            inputs = FilterFeatures(columns_to_filter)(inputs)
        outputs = tf.nest.map_structure(self, inputs)

        return outputs

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, tags=None, tags_to_filter=None, **kwargs):
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        return cls.from_features(column_group.columns, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        return features >> cls(**kwargs)


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


class AsSparseLayer(TabularLayer):
    def call(self, inputs, **kwargs):
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()
            else:
                outputs[name] = val

        return outputs


class AsDenseLayer(TabularLayer):
    def call(self, inputs, **kwargs):
        outputs = {}
        for name, val in inputs.items():
            if isinstance(val, tuple):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                outputs[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor()
            else:
                outputs[name] = val

        return outputs


class ParseTokenizedText(TabularLayer):
    def call(self, inputs, **kwargs):
        outputs, text_tensors, text_column_names = {}, {}, []
        for name, val in inputs.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                text_tensors[name] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor()
                text_column_names.append("/".join(name.split("/")[:-1]))
            # else:
            #     outputs[name] = val

        for text_col in set(text_column_names):
            outputs[text_col] = dict(input_ids=tf.cast(text_tensors[text_col + "/tokens"], tf.int32),
                                     attention_mask=tf.cast(text_tensors[text_col + "/attention_mask"], tf.int32))

        return outputs


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
