import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

_INTERACTION_TYPES = (None, "field_all", "field_each", "field_interaction")


class DotProductInteraction(tf.keras.layers.Layer):
    """
    Layer implementing the factorization machine style feature
    interaction layer suggested by the DLRM and DeepFM architectures,
    generalized to include a dot-product version of the parameterized
    interaction suggested by the FiBiNet architecture (which normally
    uses element-wise multiplication instead of dot product). Maps from
    tensors of shape `(batch_size, num_features, embedding_dim)` to
    tensors of shape `(batch_size, (num_features - 1)*num_features // 2)`
    if `self_interaction` is `False`, otherwise `(batch_size, num_features**2)`.
    Parameters
    ------------------------
    interaction_type: {}
        The type of feature interaction to use. `None` defaults to the
        standard factorization machine style interaction, and the
        alternatives use the implementation defined in the FiBiNet
        architecture (with the element-wise multiplication replaced
        with a dot product).
    self_interaction: bool
        Whether to calculate the interaction of a feature with itself.
    """.format(
        _INTERACTION_TYPES
    )

    def __init__(self, interaction_type=None, self_interaction=False, name=None, **kwargs):
        if interaction_type not in _INTERACTION_TYPES:
            raise ValueError("Unknown interaction type {}".format(interaction_type))
        self.interaction_type = interaction_type
        self.self_interaction = self_interaction
        super(DotProductInteraction, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.interaction_type is None:
            self.built = True
            return

        kernel_shape = [input_shape[2], input_shape[2]]
        if self.interaction_type in _INTERACTION_TYPES[2:]:
            idx = _INTERACTION_TYPES.index(self.interaction_type)
            for _ in range(idx - 1):
                kernel_shape.insert(0, input_shape[1])

        self.kernel = self.add_weight(
            name="bilinear_interaction_kernel",
            shape=kernel_shape,
            initializer="glorot_normal",
            trainable=True,
        )
        self.built = True

    def call(self, input):
        right = input

        # first transform v_i depending on the interaction type
        if self.interaction_type is None:
            left = input
        elif self.interaction_type == "field_all":
            left = tf.matmul(input, self.kernel)
        elif self.interaction_type == "field_each":
            left = tf.einsum("b...k,...jk->b...j", input, self.kernel)
        else:
            left = tf.einsum("b...k,f...jk->bf...j", input, self.kernel)

        # do the interaction between v_i and v_j
        # output shape will be (batch_size, num_features, num_features)
        if self.interaction_type != "field_interaction":
            interactions = tf.matmul(left, right, transpose_b=True)
        else:
            interactions = tf.einsum("b...jk,b...k->b...j", left, right)

        # mask out the appropriate area
        ones = tf.reduce_sum(tf.zeros_like(interactions), axis=0) + 1
        mask = tf.linalg.band_part(ones, 0, -1)  # set lower diagonal to zero
        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)  # get rid of diagonal
        mask = tf.cast(mask, tf.bool)
        x = tf.boolean_mask(interactions, mask, axis=1)

        # masking destroys shape information, set explicitly
        x.set_shape(self.compute_output_shape(input.shape))
        return x

    def compute_output_shape(self, input_shape):
        if self.self_interaction:
            output_dim = input_shape[1] ** 2
        else:
            output_dim = input_shape[1] * (input_shape[1] - 1) // 2
        return (input_shape[0], output_dim)

    def get_config(self):
        return {
            "interaction_type": self.interaction_type,
            "self_interaction": self.self_interaction,
        }


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col.name)


def _validate_numeric_column(feature_column):
    if len(feature_column.shape) > 1:
        raise ValueError(
            "Matrix numeric features are not allowed, "
            "found feature {} with shape {}".format(feature_column.key, feature_column.shape)
        )
    elif feature_column.shape[0] != 1:
        raise ValueError(
            "Vector numeric features are not allowed, "
            "found feature {} with shape {}".format(feature_column.key, feature_column.shape)
        )


def _validate_categorical_column(feature_column):
    if not isinstance(feature_column, fc.IdentityCategoricalColumn):
        raise ValueError(
            "Only acceptable categorical columns for feeding "
            "embeddings are identity, found wrong column type {}. "
            "Consider using NVTabular online preprocessing to perform "
            "categorical transformations".format(type(feature_column))
        )


def _validate_stack_dimensions(feature_columns):
    dims = []
    for feature_column in feature_columns:
        try:
            dimension = feature_column.dimension
        except AttributeError:
            dimension = feature_column.shape[0]
        dims.append(dimension)

    dim0 = dims[0]
    if not all([dim == dim0 for dim in dims[1:]]):
        raise ValueError(
            "'stack' aggregation requires all categorical "
            "embeddings and continuous features to have same "
            "size. Found dimensions {}".format(dims)
        )


def _validate_feature_columns(feature_columns):
    # TODO: check everything and then raise errors at the end
    for feature_column in feature_columns:
        if isinstance(feature_column, fc.CategoricalColumn):
            # TODO: I think this technically excludes BucketizedColumns,
            # which could in theory be treated numerically
            raise ValueError(
                "All feature columns must be dense! Please wrap "
                "categorical columns in embedding or indicator "
                "columns before passing"
            )
        elif isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _validate_categorical_column(feature_column.categorical_column)

        elif isinstance(feature_column, fc.NumericColumn):
            _validate_numeric_column(feature_column)


class ScalarDenseFeatures(tf.keras.layers.Layer):
    """
    Layer which maps scalar numeric and one-hot categorical
    features to a dense embedding.
    """

    def __init__(self, feature_columns, aggregation="concat", name=None, **kwargs):
        # sort feature columns to make layer independent of column order
        feature_columns = _sort_columns(feature_columns)
        _validate_feature_columns(feature_columns)

        assert aggregation in ("concat", "stack")
        if aggregation == "stack":
            _validate_stack_dimensions(feature_columns)

        self.feature_columns = feature_columns
        self.aggregation = aggregation
        super(ScalarDenseFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all([shape[1] == 1 for shape in input_shapes.values()])

        self.embedding_tables = {}
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                continue

            feature_name = feature_column.categorical_column.key
            num_buckets = feature_column.categorical_column.num_buckets
            if isinstance(feature_column, fc.EmbeddingColumn):
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=True,
                    initializer="glorot_normal",
                    shape=(num_buckets, feature_column.dimension),
                )
            else:
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=False,
                    initializer=tf.constant_initializer(np.eye(num_buckets)),
                    shape=(num_buckets, num_buckets),
                )
        self.built = True

    def call(self, inputs):
        features = []
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                features.append(inputs[feature_column.name])
            else:
                feature_name = feature_column.categorical_column.name
                table = self.embedding_tables[feature_name]
                embeddings = tf.gather(table, inputs[feature_name][:, 0])
                features.append(embeddings)

        if self.aggregation == "stack":
            return tf.stack(features, axis=1)
        return tf.concat(features, axis=1)

    def compute_output_shape(self, input_shapes):
        input_shape = [i for i in input_shapes.values()][0]
        if self.aggregation == "concat":
            output_dim = len(self.numeric_features) + sum(
                [shape[-1] for shape in self.embedding_shapes.values()]
            )
            return (input_shape[0], output_dim)
        else:
            embedding_dim = [i for i in self.embedding_shapes.values()][0]
            return (input_shape[0], len(self.embedding_shapes), embedding_dim)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
            "aggregation": self.aggregation,
        }
