Accelerated Training with TensorFlow
====================================

When training pipelines with TensorFlow, the dataloader cannot prepare
sequential batches fast enough, so the GPU is not fully utilized. To
combat this issue, we’ve developed a highly customized tabular
dataloader, ``KerasSequenceLoader``, to accelerate existing pipelines in
TensorFlow. In our experiments, we were able to achieve a speed-up 9
times as fast as the same training workflow that contains a NVTabular
dataloader. The NVTabular dataloader is capable of:

-  removing bottlenecks from dataloading by processing large chunks of
   data at a time instead of item by item
-  processing datasets that don’t fit within the GPU or CPU memory by
   streaming from the disk
-  reading data directly into the GPU memory and removing CPU-GPU
   communication
-  preparing batch asynchronously into the GPU to avoid CPU-GPU
   communication
-  supporting commonly used formats such as parquet
-  integrating easily into existing TensorFlow training pipelines by
   using a similar API as the native TensorFlow dataloader since it
   works with tf.keras models

When ``KerasSequenceLoader`` accelerates training with TensorFlow, the
following happens:

1. The required libraries are imported. The dataloader loads and
   prepares batches directly in the GPU and requires some of the GPU
   memory. Before initializing TensorFlow, the amount of memory that is
   allocated to TensorFlow needs to be controlled as well as the
   remaining memory allocation that is allocated to the dataloader. The
   environment variable 'TF\_MEMORY\_ALLOCATION' can be used to control
   the TensorFlow memory allocation.

  .. code:: python

    import tensorflow as tf

    # Control how much memory to give TensorFlow with this environment variable
    # IMPORTANT: Do this before you initialize the TensorFlow runtime, otherwise
    # it's too late and TensorFlow will claim all free GPU memory
    os.environ['TF_MEMORY_ALLOCATION'] = "8192" # explicit MB
    os.environ['TF_MEMORY_ALLOCATION'] = "0.5" # fraction of free memory
    from nvtabular.loader.tensorflow import KerasSequenceLoader,
    KerasSequenceValidater

2. The data schema is defined with ``tf.feature_columns``, the
   categorical input features (``CATEGORICAL_COLUMNS``) are fed through
   an embedding layer, and the continuous input (``CONTINUOUS_COLUMNS``)
   features are defined with ``numeric_column``. The
   ``EMBEDDING_TABLE_SHAPES`` is a dictionary that contains cardinality
   and emb\_size tuples for each categorical feature.

  .. code:: python

    def make_categorical_embedding_column(name, dictionary_size, embedding_dim):
        return tf.feature_column.embedding_column(
           tf.feature_column.categorical_column_with_identity(name, dictionary_size),
               embedding_dim
        )

    # instantiate the columns
    categorical_columns = [
       make_categorical_embedding_column(name,*EMBEDDING_TABLE_SHAPES[name]) for name in CATEGORICAL_COLUMNS
    ]
    continuous_columns = [
       tf.feature_column.numeric_column(name, (1,)) for name in CONTINUOUS_COLUMNS
    ]

3. The NVTabular dataloader is initialized. The NVTabular dataloader
   supports a list of filenames and glob pattern as input, which it will
   load and iterate over. ``feature_columns`` defines the data
   structure, which uses the ``tf.feature_column`` structure that was
   previously defined. The\ ``batch_size``, ``label_names`` (target
   columns), ``shuffle``, and ``buffer_size`` are defined.

  .. code:: python

    TRAIN_PATHS = glob.glob("./train/*.parquet")
    train_dataset_tf = KerasSequenceLoader(
       TRAIN_PATHS, # you could also use a glob pattern
       feature_columns=categorical_columns + continuous_columns,
       batch_size=BATCH_SIZE,
       label_names=LABEL_COLUMNS,
       shuffle=True,
       buffer_size=0.06  # amount of data, as a fraction of GPU memory, to load at one time
    )

4. The TensorFlow Keras model ( ``tf.keras.Model``) is defined if a
   neural network architecture is created in which ``inputs`` are the
   input tensors and ``output`` is the output tensors.

  .. code:: python

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile('sgd', 'binary_crossentropy')

5. The model is trained with ``model.fit`` using the NVTabular
   dataloader.

  .. code:: python
  
    history = model.fit(train_dataset_tf, epochs=5)

**Note**: If using the NVTabular dataloader for the validation dataset,
a callback can be used for it.

  .. code:: python

    valid_dataset_tf = KerasSequenceLoader(...)
    validation_callback = KerasSequenceValidater(valid_dataset_tf)
    history = model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=5)

You can find additional examples in our repository such as
`MovieLens <../examples/getting-started-movielens/>`__ and
`Outbrain <../examples/advanced-ops-outbrain/>`__.
