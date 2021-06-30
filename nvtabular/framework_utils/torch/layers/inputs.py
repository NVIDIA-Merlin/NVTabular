from typing import Optional, Callable, Text, Any


class TableConfig(object):
    """Configuration data for one embedding table.
    This class holds the configuration data for a single embedding table. It is
    used as the `table` parameter of a
    `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
    `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
    `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
    table will be created for those feature lookups.
    ```python
    table_config_one = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=...,
        dim=...)
    table_config_two = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=...,
        dim=...)
    feature_config = {
        'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_one),
        'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_one),
        'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_two)}
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        batch_size=...
        optimizer=tf.tpu.experimental.embedding.Adam(0.1))
    ```
    The above configuration has 2 tables, and three features. The first two
    features will be looked up in the first table and the third feature will be
    looked up in the second table.
    """

    def __init__(self,
                 vocabulary_size: int,
                 dim: int,
                 initializer: Optional[Callable[[Any], None]],
                 optimizer: Optional[_Optimizer] = None,
                 combiner: Text = "mean",
                 name: Optional[Text] = None):
        """Embedding table configuration.
        Args:
          vocabulary_size: Size of the table's vocabulary (number of rows).
          dim: The embedding dimension (width) of the table.
          initializer: A callable initializer taking one parameter, the shape of the
            variable that will be initialized. Will be called once per task, to
            initialize that task's shard of the embedding table. If not specified,
            defaults to `truncated_normal_initializer` with mean `0.0` and standard
            deviation `1/sqrt(dim)`.
          optimizer: An optional instance of an optimizer parameters class, instance
            of one of `tf.tpu.experimental.embedding.SGD`,
            `tf.tpu.experimental.embedding.Adagrad` or
            `tf.tpu.experimental.embedding.Adam`. It set will override the global
            optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.
          combiner: A string specifying how to reduce if there are multiple entries
            in a single row. Currently 'mean', 'sqrtn', 'sum' are supported, with
            'mean' the default. 'sqrtn' often achieves good accuracy, in particular
            with bag-of-words columns. For more information, see
            `tf.nn.embedding_lookup_sparse`.
          name: An optional string used to name the table. Useful for debugging.
        Returns:
          `TableConfig`.
        Raises:
          ValueError: if `vocabulary_size` is not a positive integer.
          ValueError: if `dim` is not a positive integer.
          ValueError: if `initializer` is specified and is not callable.
          ValueError: if `combiner` is not supported.
        """
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("Invalid vocabulary_size {}.".format(vocabulary_size))

        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Invalid dim {}.".format(dim))

        if (initializer is not None) and (not callable(initializer)):
            raise ValueError("initializer must be callable if specified.")
        if initializer is None:
            initializer = init_ops_v2.TruncatedNormal(mean=0.0,
                                                      stddev=1 / math.sqrt(dim))

        if combiner not in ("mean", "sum", "sqrtn"):
            raise ValueError("Invalid combiner {}".format(combiner))

        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.initializer = initializer
        self.optimizer = optimizer
        self.combiner = combiner
        self.name = name

    def __repr__(self):
        # If using the default initializer, just print "None" for clarity.
        initializer = self.initializer

        if isinstance(initializer, init_ops_v2.TruncatedNormal):
            # PY2 type checking can't infer type of initializer even after if.
            initializer = typing.cast(init_ops_v2.TruncatedNormal, initializer)
            if (initializer.mean == 0.0
                    and math.isclose(initializer.stddev,
                                     1 / math.sqrt(self.dim))):  # pytype: disable=module-attr (math.isclose not in PY2)
                initializer = None

        return (
            "TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, "
            "initializer={initializer!r}, optimizer={optimizer!r}, "
            "combiner={combiner!r}, name={name!r})".format(
                vocabulary_size=self.vocabulary_size,
                dim=self.dim,
                initializer=initializer,
                optimizer=self.optimizer,
                combiner=self.combiner,
                name=self.name, )
        )


class FeatureConfig(object):
    """Configuration data for one embedding feature.
    This class holds the configuration data for a single embedding feature. The
    main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
    via the table parameter:
    ```python
    table_config_one = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=...,
        dim=...)
    table_config_two = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=...,
        dim=...)
    feature_config = {
        'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_one),
        'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_one),
        'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
            table=table_config_two)}
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        batch_size=...
        optimizer=tf.tpu.experimental.embedding.Adam(0.1))
    ```
    The above configuration has 2 tables, and three features. The first two
    features will be looked up in the first table and the third feature will be
    looked up in the second table.
    When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
    `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
    `max_sequence_length` is 0, the default, you should expect a output of
    `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
    `max_sequence_length` is greater than 0, the feature is embedded as a sequence
    and padded up to the given length. The shape of the output for this feature
    will be `(batch_size, max_sequence_length, dim)`.
    """

    def __init__(self,
                 table: TableConfig,
                 max_sequence_length: int = 0,
                 name: Optional[Text] = None):
        """Feature configuration.
        Args:
          table: An instance of `tf.tpu.experimental.embedding.TableConfig`,
            describing the table in which this feature should be looked up.
          max_sequence_length: If positive, the feature is a sequence feature with
            the corresponding maximum sequence length. If the sequence is longer
            than this, it will be truncated. If 0, the feature is not a sequence
            feature.
          name: An optional name for the feature, useful for debugging.
        Returns:
          `FeatureConfig`.
        Raises:
          ValueError: if `table` is not an instance of
            `tf.tpu.experimental.embedding.TableConfig`.
          ValueError: if `max_sequence_length` not an integer or is negative.
        """
        if not isinstance(table, TableConfig):
            raise ValueError("table is type {}, expected "
                             "`tf.tpu.experimental.embedding.TableConfig`".format(
                type(table)))

        if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
            raise ValueError("Invalid max_sequence_length {}.".format(
                max_sequence_length))

        self.table = table
        self.max_sequence_length = max_sequence_length
        self.name = name

    def __repr__(self):
        return (
            "FeatureConfig(table={table!r}, "
            "max_sequence_length={max_sequence_length!r}, name={name!r})"
                .format(
                table=self.table,
                max_sequence_length=self.max_sequence_length,
                name=self.name)
        )
