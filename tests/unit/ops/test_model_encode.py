#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from functools import partial

import pandas as pd
import pytest

import nvtabular as nvt
from nvtabular.dispatch import HAS_GPU

if HAS_GPU:
    _CPU = [True, False]
    _HAS_GPU = True
else:
    _CPU = [True]
    _HAS_GPU = False


class SelectionModel:
    # Simple model that returns a "selected"
    # column of an input Array
    # (when SelectionModel.predict is called)
    def __init__(self, j):
        self.j = j

    def predict(self, x):
        # Fake prediction just returns column `j`
        return x[:, self.j]

    @classmethod
    def load_model(cls, model_str):
        # Mimic model loading by
        # initializing a new SelectionModel
        # object using the integer at the
        # very end of `model_str`
        j = int(model_str.split("_")[-1])
        return cls(j)


def simple_iterator(df):
    # Simple iterator function that returns
    # a list of numpy/cupy arrays. Note that
    # each element is an (array, str) tuple
    # so that `model_encode_func` is required
    # for proper behavior
    arr = df.values
    batch_size = 3
    return [(arr[i : i + batch_size, :], "filler") for i in range(0, arr.shape[0], batch_size)]


def simple_encode(x, y):
    return x.predict(y[0])


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
@pytest.mark.parametrize("cpu", _CPU)
def test_model_encode(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, cpu):

    # Define ModelEncode Operator
    model = "SelectionModel_0"  # Mimic model loading below
    output_names = "prediction"  # Name of column being added
    data_iterator_func = simple_iterator
    model_load_func = SelectionModel.load_model
    model_encode_func = simple_encode
    output_concat_func = None  # Result is already numpy/cupy
    features = op_columns >> nvt.ops.ModelEncode(
        model,
        output_names,
        data_iterator_func=data_iterator_func,
        model_load_func=model_load_func,
        model_encode_func=model_encode_func,
        output_concat_func=output_concat_func,
    )

    #   NOTES:
    # - For "real" data, `data_iterator_func` will likely
    #   correspond to a `Dataloader`` initializer.
    # - The purpose of `model_encode_func` is to deal with
    #   the fact that `model(batch)` may not be the correct
    #   syntax for performing a prediction. This is because
    #   you may want to select a specific index of `batch`,
    #   and you may want to use a specific attribute of `model`
    # - In this case, `output_concat_func` did not need to
    #   be specified, because the output format is already
    #   numpy or cupy.

    # Fit and transform
    processor = nvt.Workflow(features)
    ds = nvt.Dataset(dataset.to_ddf(), cpu=cpu)
    df0 = df
    if cpu and not isinstance(df0, pd.DataFrame):
        df0 = df0.to_pandas()
    processor.fit(ds)
    new_df = processor.transform(ds).to_ddf().compute()

    # Check that ModelEncode worked as expected
    assert new_df[output_names].all() == new_df[op_columns[0]].all()


def simple_tf_model(cat_names=None, cont_names=None):
    import tensorflow as tf

    # instantiate our columns
    categorical_columns = [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                name,
                1000,  # dictionary_size
            ),
            10,  # embedding_dim
        )
        for name in cat_names
    ]

    # Categorical
    categorical_inputs = {}
    for column_name in cat_names:
        categorical_inputs[column_name] = tf.keras.Input(
            name=column_name, shape=(1,), dtype=tf.int64
        )
    categorical_embedding_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    categorical_x = categorical_embedding_layer(categorical_inputs)

    # Continuous
    continuous_inputs = []
    for column_name in cont_names:
        continuous_inputs.append(tf.keras.Input(name=column_name, shape=(1,), dtype=tf.float32))
    continuous_embedding_layer = tf.keras.layers.Concatenate(axis=1)
    continuous_x = continuous_embedding_layer(continuous_inputs)
    continuous_x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(continuous_x)

    # Concatenate and build MLP
    x = tf.keras.layers.Concatenate(axis=1)([categorical_x, continuous_x])
    for dim in [8, 4]:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = tf.keras.layers.Dense(1, activation="linear")(x)
    inputs = list(categorical_inputs.values()) + continuous_inputs
    return tf.keras.Model(inputs=inputs, outputs=x)


def simple_tf_encode(model, batch):
    return model(batch[0]).numpy()


def simple_tf_iterator(
    data,
    dataloader=None,
    cpu=None,
    batch_size=None,
    cat_names=None,
    cont_names=None,
):
    return dataloader(
        nvt.Dataset(data, cpu=cpu),
        batch_size=batch_size,
        cat_names=cat_names,
        cont_names=cont_names,
        label_names=[],
    )


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("batch_size", [2, 32])
@pytest.mark.parametrize("cpu", _CPU)
def test_tf_model_encode(tmpdir, df, dataset, gpu_memory_frac, engine, batch_size, cpu):

    # Skip this test if tensorflow deps are missing.
    # TODO: Import the dataloader from Merlin-Models?
    os.environ["TF_MEMORY_ALLOCATION"] = "0.1"
    merlin_tf = pytest.importorskip("nvtabular.loader.tensorflow")
    tf_models = pytest.importorskip("tensorflow.keras.models")

    # Define a "toy" model
    # TODO: Improve this model?
    model_path = str(tmpdir) + "/" + "model.h5"
    cat_names = ["id"]
    cont_names = ["x", "y"]
    op_columns = cat_names + cont_names
    tf_model = simple_tf_model(cat_names=cat_names, cont_names=cont_names)
    tf_model.save(model_path)

    # Define a "toy" dataloader
    dataloader = merlin_tf.KerasSequenceLoader
    dataloader_kwargs = {
        "cat_names": cat_names,
        "cont_names": cont_names,
        "batch_size": batch_size,
    }

    # Define ModelEncode Operator
    features = op_columns >> nvt.ops.ModelEncode(
        model_path,
        "prediction",
        data_iterator_func=partial(
            simple_tf_iterator,
            dataloader=dataloader,
            cpu=cpu,
            **dataloader_kwargs,
        ),
        model_load_func=tf_models.load_model,
        model_encode_func=simple_tf_encode,
    )

    # Fit and transform
    processor = nvt.Workflow(features)
    ds = nvt.Dataset(dataset.to_ddf(), cpu=cpu)
    df0 = df
    if cpu and not isinstance(df0, pd.DataFrame):
        df0 = df0.to_pandas()
    processor.fit(ds)
    processor.transform(ds).to_ddf().compute(scheduler="synchronous")
