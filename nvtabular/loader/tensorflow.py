#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import os

import tensorflow as tf

from ..io import Dataset
from ..workflow import BaseWorkflow
from .backend import AsyncIterator, TensorBatchDatasetItr
from .tf_utils import configure_tensorflow, get_dataset_schema_from_feature_columns

from_dlpack = configure_tensorflow()


def _validate_dataset(paths_or_dataset, buffer_size, batch_size, engine, reader_kwargs):
    # if a dataset was passed, just return it
    if isinstance(paths_or_dataset, Dataset):
        return paths_or_dataset

    # otherwise initialize a dataset
    # from paths or glob pattern
    if isinstance(paths_or_dataset, str):
        files = tf.io.gfile.glob(paths_or_dataset)
        _is_empty_msg = "Couldn't find file pattern {} in directory {}".format(
            *os.path.split(paths_or_dataset)
        )
    else:
        # TODO: some checking around attribute
        # error here?
        files = list(paths_or_dataset)
        _is_empty_msg = "paths_or_dataset list must contain at least one filename"

    assert isinstance(files, list)
    if len(files) == 0:
        raise ValueError(_is_empty_msg)

    # implement buffer size logic
    # TODO: IMPORTANT
    # should we divide everything by 3 to account
    # for extra copies laying around due to asynchronicity?
    if buffer_size >= 1:
        if buffer_size < batch_size:
            reader_kwargs["batch_size"] = int(batch_size * buffer_size)
        else:
            reader_kwargs["batch_size"] = buffer_size
    else:
        reader_kwargs["part_mem_fraction"] = buffer_size

    return Dataset(files, engine=engine, **reader_kwargs)


def _validate_schema(feature_columns, cont_names, cat_names):
    _uses_feature_columns = feature_columns is not None
    _uses_explicit_schema = (cat_names is not None) or (cont_names is not None)
    if _uses_feature_columns and _uses_explicit_schema:
        raise ValueError(
            "Passed `feature_column`s and explicit column names, must be one or the other"
        )
    elif _uses_feature_columns:
        return get_dataset_schema_from_feature_columns(feature_columns)
    elif _uses_explicit_schema:
        cat_names = cat_names or []
        cont_names = cont_names or []
        return cat_names, cont_names
    else:
        raise ValueError(
            "Must either pass a list of TensorFlow `feature_column`s "
            "or explicit `cat_name` and `cont_name` column name lists."
        )


def _validate_workflows(workflows, cat_names, cont_names, label_name):
    assert all([isinstance(w, BaseWorkflow) for w in workflows])
    for workflow in workflows:
        assert workflow.columns_ctx["categorical"]["base"] == cat_names
        assert workflow.columns_ctx["continuous"]["base"] == cont_names
        assert workflow.columns_ctx["label"]["base"] == [label_name]

        cat_names = workflow.columns_ctx["final"]["categorical"]
        cont_names = workflow.columns_ctx["final"]["continuous"]
        label_name = workflow.columns_ctx["final"]["label"][0]
    return workflows


class TensorFlowBatchDatasetItr(TensorBatchDatasetItr):
    def device_ctx(self, dev):
        return tf.device("/device:GPU:{}".format(dev))

    def _to_tensor(self, gdf, dtype=None):
        if gdf.empty:
            return
        dlpack = self.to_dlpack(gdf)
        x = from_dlpack(dlpack)
        # TODO: type checking?
        return tf.expand_dims(x, -1)

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        # TODO: can we use these somehow to go faster?
        # what's the cost of doing axis 1 slicing in TF?
        # gdf_cats, gdf_conts, gdf_label = (
        #     gdf[cat_names], gdf[cont_names], gdf[label_names]
        # )
        X = {}
        for name in cat_names + cont_names + label_names:
            X[name] = self._to_tensor(gdf.pop(name))
        del gdf
        return X


class KerasSequenceLoader(tf.keras.utils.Sequence):
    _itr_cls = TensorFlowBatchDatasetItr

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
        workflows=None,
        reader_kwargs={},
    ):
        self.data = _validate_dataset(
            paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs
        )
        self.cat_names, self.cont_names = _validate_schema(feature_columns, cat_names, cont_names)
        self.label_name = label_name

        # TODO: do we even need to save most of the attributes or
        # can we do a functools.partial on AsyncIterator?
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._itr = None

        workflows = workflows or []
        self.workflows = _validate_workflows(
            workflows, self.cat_names, self.cont_names, self.label_name
        )

    def __len__(self):
        """
        should be "global" to all data loaders
        """
        return math.ceil(self.data.num_rows // self.batch_size)

    def __iter__(self):
        """
        should be "global" to all data loaders
        """
        return iter(
            AsyncIterator(
                self._itr_cls,
                cats=self.cat_names,
                conts=self.cont_names,
                labels=[self.label_name],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                devices=None,  # TODO: support multi-gpu
            )
        )

    def map(self, workflow):
        """
        should be "global" to all data loaders
        """
        workflows = self.workflows + [workflow]
        self.workflows = _validate_workflows(
            workflows, self.cat_names, self.cont_names, self.label_name
        )
        # TODO: return copy for consistency?

    def __getitem__(self, idx):
        """
        implemented exclusively for consistency
        with Keras model.fit. Does not leverage
        passed idx in any way
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
        # TODO: does this get done before
        # or after validation?
        self._itr = None
