import os
import warnings

import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

from ..io import Dataset, _shuffle_gdf, device_mem_size
from ..workflow import BaseWorkflow
from .backend import AsyncIterator, TensorBatchDatasetItr
from .tf_utils import configure_tensorflow


configure_tensorflow()


class TensorFlowBatchDatasetItr(TensorBatchDatasetItr):
    # TODO: fill in
    pass


class KerasSequenceLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        paths_or_dataset,
        columns,
        batch_size,
        label_name,
        engine=None,
        shuffle=True,
        buffer_size=10,
        dataset_size=None,
        reader_kwargs={},
    ):

        construct_iter = True
        if isinstance(paths_or_dataset, Dataset):
            construct_iter = False
        else:
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

    def _initialize_iterator(self):
        itr = TensorFlowBatchDatasetItr(self.dataset)
        self._itr = iter(AsyncIterator(
            itr,
            cats=self.cats,
            conts=self.conts,
            labels=self.labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle
            library="tensorflow"
        ))

    def __iter__(self):
        if self._itr is None:
            self._initialize_iterator()
        return self._itr

    def __getitem__(self, idx):
        '''
        implemented exclusively for consistency
        with Keras model.fit. Does not leverage
        passed in any way
        '''
        # TODO: add in checks on idx increments to ensure
        # that the user isn't expecting any functionality
        # that isn't there?
        return self.__next__()

    def __next__(self):
        if self._itr is None:
            self._initialize_iterator()
        return next(self._itr)

    def on_epoch_end(self):
        # this way we know to reinitialize
        self.itr = None