import os
import warnings

import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

from ..io import Dataset, _shuffle_gdf, device_mem_size
from ..workflow import BaseWorkflow
from .backend import AsyncIterator, TensorBatchDatasetItr
from .tf_utils import configure_tensorflow


configure_tensorflow()


def _validate_dataset(paths_or_dataset, buffer_size, batch_size, engine, reader_kwargs):
    if isinstance(paths_or_dataset, Dataset):
        return paths_or_dataset

    # use tf glob to find matching files
    if isinstance(paths_or_dataset, str):
        files = tf.io.gfile.glob(paths_or_dataset)
        _is_empty_msg = "Couldn't find file pattern {} in directory {}".format(
            *os.path.split(paths_or_dataset)
        )
    else:
        files = paths_or_dataset
        _is_empty_msg = "paths_or_dataset list must contain at least one filename"

    assert isinstance(files, list)
    if len(files) == 0:
        raise ValueError(_is_empty_msg)

    if buffer_size >= 1:
        if buffer_size < batch_size:
            reader_kwargs["batch_size"] = int(batch_size * buffer_size)
        else:
            reader_kwargs["batch_size"] = buffer_size
    else:
        reader_kwargs["part_mem_fraction"] = buffer_size

    return Dataset(files, engine=engine, **reader_kwargs)


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


def _get_data_schema_from_columns(columns):
    # TODO: add checking and warning about columns
    # that leverage transformations and suggest
    # NVT equivalent?
    base_columns = set()
    for column in columns:
        base_columns |= _get_parents(column)

    cat_names, cont_names = [], []
    for column in base_columns:
        if isinstance(column, fc.CategoricalColumn):
            cat_names.append(column.name)
        else:
            cont_names.append(column.name)
    return cat_names, cont_names


def _validate_schema(feature_columns, cont_names, cat_names):
    _uses_feature_columns = feature_columns is not None
    _uses_explicit_schema = (cat_names is not None) or (cont_names is not None)
    if _uses_feature_columns and _uses_explicit_schema:
        raise ValueError(
            "Passed `feature_column`s and explicit column " "names, must be one or the other"
        )
    elif _uses_feature_columns:
        return _get_data_schema_from_columns(feature_columns)
    elif _uses_explicit_schema:
        cat_names = cat_names or []
        cont_names = cont_names or []
        return cat_names, cont_names
    else:
        raise ValueError(
            "Must either pass a list of TensorFlow `feature_column`s "
            "or explicit `cat_name` and `cont_name` column name lists."
        )


class TensorFlowBatchDatasetItr(TensorBatchDatasetItr):
    # TODO: fill in
    pass


class KerasSequenceLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        label_name,
        feature_columns=None,
        cat_names=None,
        cont_names=None,
        engine=None,
        shuffle=True,
        buffer_size=0.1,
        reader_kwargs={},
    ):
        self.dataset = _validate_dataset(
            paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs
        )
        self.cat_names, self.cont_names = _validate_schema(feature_columns, cat_names, cont_names)
        self.label_name = label_name

        # TODO: do we even need to save most of the attributes or
        # can we do a functools.partial on AsyncIterator?
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._itr = None

    def __iter__(self):
        itr = TensorFlowBatchDatasetItr(self.dataset)
        return iter(
            AsyncIterator(
                itr,
                cats=self.cat_names,
                conts=self.cont_names,
                labels=[self.label_name],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                library="tensorflow",
            )
        )

    def __getitem__(self, idx):
        """
        implemented exclusively for consistency
        with Keras model.fit. Does not leverage
        passed in any way
        """
        # TODO: add in checks on idx increments to ensure
        # that the user isn't expecting any functionality
        # that isn't there?
        return self.__next__()

    def __next__(self):
        self._itr = self._itr or iter(self)
        return next(self._itr)

    def on_epoch_end(self):
        # this way we know to reinitialize
        self._itr = None
