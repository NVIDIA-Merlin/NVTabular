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
import os

import tensorflow as tf

from nvtabular.io import Dataset
from nvtabular.loader.backend import DataLoader
from nvtabular.loader.tf_utils import configure_tensorflow, get_dataset_schema_from_feature_columns

from_dlpack = configure_tensorflow()


def _validate_dataset(paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs):
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


class KerasSequenceLoader(tf.keras.utils.Sequence, DataLoader):
    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        label_names,
        feature_columns=None,
        cat_names=None,
        cont_names=None,
        engine=None,
        shuffle=True,
        buffer_size=0.1,
        workflows=None,
        devices=None,
        parts_per_chunk=1,
        reader_kwargs={},
    ):
        dataset = _validate_dataset(
            paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs
        )
        cat_names, cont_names = _validate_schema(feature_columns, cat_names, cont_names)

        DataLoader.__init__(
            self,
            dataset,
            cat_names,
            cont_names,
            label_names,
            batch_size,
            shuffle,
            parts_per_chunk=parts_per_chunk,
            workflows=workflows,
            devices=None,  # TODO: figure out multi-gpu support
        )

    def __len__(self):
        """
        recreating since otherwise Keras yells at you
        """
        # TODO: what's a better way to do this inheritance
        # of the appropriate methods? A Metaclass?
        return DataLoader.__len__(self)

    def __getitem__(self, idx):
        """
        implemented exclusively for consistency
        with Keras model.fit. Does not leverage
        passed idx in any way
        """
        try:
            return DataLoader.__next__(self)
        except StopIteration:
            # TODO: I would like to do a check for idx == 0
            # here, but that requires that tf.keras.Model.fit
            # be called with shuffle=False, and that seems
            # small enough that it would be too easy to miss
            # for many users. That said, blind reinitialization
            # is probably irresponsible, so worth thinking
            # of something better here
            DataLoader.__iter__(self)
            return DataLoader.__next__(self)

    def _get_device_ctx(self, dev):
        return tf.device("/device:GPU:{}".format(dev))

    def _to_tensor(self, gdf, dtype=None):
        if gdf.empty:
            return

        # TODO: file TF bug about column major stride arrays
        if gdf.shape[1] == 1:
            dlpack = gdf.to_dlpack()
        else:
            dlpack = gdf.values.T.toDlpack()
        x = from_dlpack(dlpack)
        if gdf.shape[1] > 1:
            x = tf.transpose(x)
        return x

    def _create_batch(self, tensor, num_samples):
        idx = self._get_segment_lengths(num_samples)
        return tf.split(tensor, idx)

    def _handle_tensors(self, cats, conts, labels):
        X = {}
        for tensor, names in zip([cats, conts], [self.cat_names, self.cont_names]):
            if len(names) > 1:
                tensors = tf.split(tensor, len(names), axis=1)
            else:
                tensors = [tensor]
            X.update({name: x for name, x in zip(names, tensors)})

        if len(self.label_names) > 1:
            labels = tf.split(labels, len(self.label_names), axis=1)
        else:
            labels = [labels]
        return X, labels


class _StreamingMetric:
    def __init__(self, name):
        self.name = name
        self.value = 0
        self.samples = 0

    def update(self, update, n):
        self.value *= self.samples / (self.samples + n)
        self.value += (n * update) / (n + self.samples)
        self.samples += n


class KerasSequenceValidater(tf.keras.callbacks.Callback):
    _supports_tf_logs = True

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def on_epoch_end(self, epoch, logs={}):
        streaming_metrics = [_StreamingMetric(name) for name in self.model.metrics_names]
        for X, y in self.dataloader:
            n = y[0].shape[0]
            scores = self.model.evaluate(X, y, batch_size=n, verbose=0)
            for metric, score in zip(streaming_metrics, scores):
                metric.update(score, n)
        for metric in streaming_metrics:
            logs["val_" + metric.name] = metric.value
        return logs
