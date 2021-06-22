import tensorflow as tf

from nvtabular.column_group import ColumnGroup


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

    def compute_output_shape(self, input_shape):
        return {k: v for k, v in input_shape.items() if k in self.columns}

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
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        if self.aggregation == "concat":
            return batch_size, sum([i[1] for i in input_shapes.values()])
        elif self.aggregation == "stack":
            last_dim = [i for i in input_shapes.values()][0][-1]

            return batch_size, len(input_shapes), last_dim

        return input_shapes

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

    def calculate_batch_size_from_input_shapes(self, input_shapes):
        return [i for i in input_shapes.values() if not isinstance(i, tuple)][0][0]

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, tags=None, tags_to_filter=None, **kwargs):
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        return cls.from_features(column_group.columns, **kwargs)

    @classmethod
    def from_features(cls, features, **kwargs):
        return features >> cls(**kwargs)

    def __rrshift__(self, other):
        from nvtabular.framework_utils.tensorflow.block import right_shift_layer

        return right_shift_layer(self, other)


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
    def __init__(self, max_text_length=None, aggregation=None, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(aggregation, trainable, name, dtype, dynamic, **kwargs)
        self.max_text_length = max_text_length

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

    def compute_output_shape(self, input_shapes):
        assert(self.max_text_length is not None)

        output_shapes, text_column_names = {}, []
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        for name, val in input_shapes.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                text_column_names.append("/".join(name.split("/")[:-1]))

        for text_col in set(text_column_names):
            output_shapes[text_col] = dict(input_ids=tf.TensorShape([batch_size, self.max_text_length]),
                                           attention_mask=tf.TensorShape([batch_size, self.max_text_length]))

        return output_shapes
