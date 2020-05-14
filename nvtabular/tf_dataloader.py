import os
import warnings

# need to configure tensorflow to not use all of memory
# TF_MEMORY_ALLOCATION is fraction of GPU memory if < 1, and size
# in MB if > 1
import rmm
import tensorflow as tf
from packaging import version
from tensorflow.python.feature_column import feature_column_v2 as fc

from .io import GPUDatasetIterator
from .workflow import Workflow, _shuffle_part

free_gpu_mem_mb = rmm.get_info().free / (1024 ** 2)
tf_mem_size = os.environ.get("TF_MEMORY_ALLOCATION", 0.5)
if float(tf_mem_size) < 1:
    tf_mem_size = free_gpu_mem_mb * float(tf_mem_size)
tf_mem_size = int(tf_mem_size)
assert tf_mem_size < free_gpu_mem_mb

tf_device = os.environ.get("TF_VISIBLE_DEVICE", 0)
try:
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices("GPU")[tf_device],
        [tf.config.LogicalDeviceConfiguration(memory_limit=tf_mem_size)],
    )
except RuntimeError:
    warnings.warn("TensorFlow runtime already initialized, may not be enough memory for cudf")


if version.parse(tf.__version__) < version.parse("2.2.0"):
    from tfdlpack import from_dlpack as tf_from_dlpack
else:
    from tensorflow.experimental.dlpack import from_dlpack as tf_from_dlpack

    warnings.warn("Tensorflow 2.2.0 dlpack integration has known memory leak issues")


def _to_tensor(x):
    """
  map a cudf series `x` to a `(len(x), 1)` dim
  TensorFlow tensor
  """
    # catch cudf warning about row ordering since
    # we're just grabbing vectors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capsule = x.to_dlpack()

    tensor = tf_from_dlpack(capsule)
    return tf.expand_dims(tensor, -1)


def _num_steps(total_number, step_size):
    """
  get the number of steps of size `step_size` in a
  collection of `total_number` items. If `step_size`
  is not a factor of `total_number`, includes the
  last smaller-sized step
  """
    return (total_number - 1) // step_size + 1


def _get_parents(column):
    """
  recursive function for finding the feature columns
  that supply inputs for a given `column`. If there are
  none, returns the column. Uses sets so is not
  deterministic.
  """
    if isinstance(column.parents[0], str):
        return set([column])
    parents = set()
    for parent in column.parents:
        parents |= _get_parents(parent)
    return parents


class KerasSequenceDataset(tf.keras.utils.Sequence):
    """
  Infinite generator used to iterate through csv or parquet dataframes
  on GPU by leveraging an NVTabular `dataset`. Applies preprocessing via
  NVTabular `Workflow`s and outputs tabular dictionaries of TensorFlow
  Tensors via dlpack and tfdlpack. Useful for training tabular models
  built in Keras, instantiated by TensorFlow Feature Columns, and trained via
  `tf.keras.Model.fit`.

  In order to minimize loading and preprocessing time, data is loaded
  in larger-than-batch-sized chunks whose exact size is decided by the
  `buffer_size` argument (see below). These chunks are loaded and
  preprocssed by any Workflows all at once to reduce bottlenecks.
  Importantly, TensorFlow's default behavior is to claim all GPU memory
  for itself, which leaves none for NVTabular to perform this
  functionality. As such, we attempt to configure TensorFlow to restrict
  its memory allocation on a given GPU using the environment variables
  `TF_MEMORY_ALLOCATION` and `TF_VISIBLE_DEVICE`. If `TF_MEMORY_ALLOCATION < 1`,
  it will be assumed that this refers to a fraction of free GPU
  memory on the given device. Otherwise, it will refer to an explicit
  allocation amount in MB. `TF_VISIBLE_DEVICE` should be an integer GPU
  index.
  
  If `shuffle` is `True`, each of chunks will be shuffled
  before being iterated through in batches of size `batch_size`.
  While this limits the randomness of batches from epoch to epoch,
  pre-randomized datasets (i.e. datasets ordered such that sequential
  reading of elements yields unbiased samples of features) and large
  buffer sizes should enable sufficient randomness to fit most tabular
  datasets of interest.

  Iterator output is of the form `(dict(features), label)`, where each
  element of the features dict is a `feature_name: feature_tensor` pair
  and all tensors are of shape `(batch_size, 1)`. Note that this means
  vectorized continuous and multi-hot categorical features are not
  currently supported, since cuDF doesn't support array-like columns.

  The underlying NVTabular `dataset` object is stored in the `nvt_dataset`
  property, and should be used for updating NVTabular `Workflow`
  statistics:
  ```python
  workflow = nvt.Workflow(...)
  dataset = KerasSequenceDataset(...)
  workflow.update_stats(dataset.nvt_dataset, record_stats=True)
  ```

  Parameters
  -------------
  - file_pattern: str or list(str)
      Either a string representing a file pattern (see `tf.glob` for
      pattern rules) or a list of filenames to be iterated through
  - columns: list(str) or list(tf.feature_column)
      Either a list of string column names to use from the dataframe(s)
      specified by `file_pattern`, or a ist of TensorFlow feature columns
      representing the inputs exposed to the model to be trained.
      Columns with parent columns will climb the parent tree, and the
      names of the columns in the unique set of terminal columns
      will be used as the column names.
  - batch_size: int
      Number of samples to yield at each iteration
  - label_name: str
      Column name of the target variable in the dataframe specified by
      `file_pattern`
  - engine: {'csv', 'parquet', None}, default None
      String specifying the type of read engine to use. If left as `None`,
      will try to infer the engine type from the file extension.
  - shuffle: bool, default True
      Whether to shuffle chunks of batches before iterating through them.
  - buffer_size: float or int
      If `0 <  buffer_size < 1`, `buffer_size` will refer to the amount of
      free GPU memory to occupy with a buffered chunk. If `1 < buffer_size <
      batch_size`, the number of rows read for a buffered chunk will
      be equal to `int(buffer_size*batch_size)`. Otherwise, if `buffer_size >
      batch_size`, `buffer_size` rows will be read in each chunk (except for
      the last chunk in a dataset, which will, in general, be smaller).
      Larger chunk sizes will lead to more efficieny and randomness,
      but require more memory.
  - dataset_size: None or int
      Number of samples in a dataset. This information will be used
      by Keras to decide when to finish an epoch and perform validation.
      Note that since we don't leverage the index provided by Keras to
      the `__getitem__` method, but keep track of batches internally,
      setting this to a value different than the actual size of the
      dataset will have no impact in the frequency with which samples
      are seen by the network. If left as `None`, initialization will
      include one iteration through the dataset to count the dataset
      size explicitly.
  """

    def __init__(
        self,
        file_pattern,
        columns,
        batch_size,
        label_name,
        engine=None,
        shuffle=True,
        buffer_size=10,
        dataset_size=None,
        reader_kwargs={},
    ):
        # use tf glob to find matching files
        if isinstance(file_pattern, str):
            files = tf.io.gfile.glob(file_pattern)
            _is_empty_msg = "Couldn't find file pattern {} in directory {}".format(
                *os.path.split(file_pattern)
            )
        else:
            files = file_pattern
            _is_empty_msg = "file_pattern list must contain at least one filename"

        assert isinstance(files, list)
        if len(files) == 0:
            raise ValueError(_is_empty_msg)

        if all([isinstance(col, fc.FeatureColumn) for col in columns]):
            # get column names by traversing parent tree of columns
            # TODO: make deterministic?
            base_columns = set()
            for column in columns:
                base_columns |= _get_parents(column)
            column_names = [column.name for column in base_columns]
        elif all([isinstance(col, str) for col in columns]):
            column_names = columns
        else:
            raise ValueError(
                "Expected columns to be either a list of all TensorFlow "
                "feature_columns or list of all strings. Got {}".format(columns)
            )

        # intialize the dataset iterator with a batch_size increased
        # by buffer_factor to do as few loads as possible
        # TODO: what's the syntax for byte range read?
        if buffer_size >= 1:
            if buffer_size < batch_size:
                reader_kwargs["batch_size"] = int(batch_size * buffer_size)
            else:
                reader_kwargs["batch_size"] = buffer_size
        else:
            reader_kwargs["gpu_memory_frac"] = buffer_size

        self._nvt_dataset = GPUDatasetIterator(
            files, columns=column_names + [label_name], engine=engine, **reader_kwargs
        )

        # get dataset size if it wasn't provided
        if dataset_size is None:
            dataset_size = 0
            for chunk in self._nvt_dataset:
                dataset_size += chunk.shape[0]
        self.dataset_size = dataset_size

        # set attributes
        self.workflows = []
        self.batch_size = batch_size
        self.label_name = label_name
        self.column_names = column_names
        self.shuffle = shuffle

        # initialize iter-relevant attributes
        self.iter_obj = None
        self.chunk_idx = 0  # determines our batch index in a chunk
        self.batches_in_chunk = None

    def __len__(self):
        return _num_steps(self.dataset_size, self.batch_size)

    @property
    def nvt_dataset(self):
        return self._nvt_dataset

    def map(self, workflow):
        """
    Map an NVTabular Workflow object onto the dataset.
    Each chunk read by the iterator will be transformed
    via `workflow.apply_ops`.
    """
        if not isinstance(workflow, Workflow):
            raise TypeError("Expected NVTabular Workflow, found type {}".format(type(workflow)))

        self.workflows.append(workflow)

    def __iter__(self):
        self._initialize_iterator()
        return self

    def _initialize_iterator(self):
        self.iter_obj = iter(self._nvt_dataset)
        self.chunk_idx = 0
        self.batches_in_chunk = None

    def _preprocess_and_shuffle(self, x):
        for workflow in self.workflows:
            x = workflow.apply_ops(x)

        if self.shuffle:
            x = _shuffle_part(x)
        return x

    def _process_next_chunk(self):
        # try to get the next chunk, intializing iterator if
        # we haven't yet and reinitializing if it's exhausted
        try:
            chunk = next(self.iter_obj)
            self.chunk_idx = 0
        except (StopIteration, TypeError):
            self._initialize_iterator()
            chunk = next(self.iter_obj)

        # first apply workflows to full chunk and shuffle if needed
        chunk = self._preprocess_and_shuffle(chunk)
        self.batches_in_chunk = _num_steps(len(chunk), self.batch_size)

        # map cuDF columns to dict of TF tensors
        # TODO: is it more efficient to do slicing in cuDF
        # then do conversion to TensorFlow at the batch level?
        tensor_map = map(
            lambda column_name: (column_name, _to_tensor(chunk[column_name])),
            self.column_names + [self.label_name],
        )

        self.inputs = {column_name: x for column_name, x in tensor_map}
        self.labels = self.inputs.pop(self.label_name)

    def __getitem__(self, idx):
        # Keras Sequence requires that this method be implemented
        # but I don't see a way (or reason frankly) to implement
        # arbitrary slicing, so we'll just call our __next__ method
        return self.__next__()

    def __next__(self):
        if self.batches_in_chunk is None or self.chunk_idx == self.batches_in_chunk:
            self._process_next_chunk()

        slc = slice(self.chunk_idx * self.batch_size, (self.chunk_idx + 1) * self.batch_size)
        x = {column_name: x[slc] for column_name, x in self.inputs.items()}
        y = self.labels[slc]

        self.chunk_idx += 1
        return (x, y)
