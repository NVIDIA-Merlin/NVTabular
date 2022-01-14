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


def simple_tf_model(input_shape=(2,)):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    return Sequential([Dense(8, input_shape=input_shape), Dense(1)])


def simple_tf_encode(model, batch):
    return model(batch).numpy()


@pytest.mark.skip
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

    # Define and save a very simple model
    model_path = tmpdir + "/" + "model.h5"
    op_columns = ["x", "y"]
    tf_model = simple_tf_model(input_shape=(len(op_columns),))
    tf_model.save(model_path)

    # Define ModelEncode Operator
    features = op_columns >> nvt.ops.ModelEncode(
        model_path,
        "prediction",
        data_iterator_func=partial(merlin_tf.KerasSequenceLoader, batch_size=batch_size),
        model_load_func=tf_models.load_model,
        model_encode_func=simple_tf_encode,
    )

    # Fit and transform
    processor = nvt.Workflow(features)
    ds = nvt.Dataset(dataset.to_ddf(), cpu=cpu)
    processor.fit(ds)
    processor.transform(ds).to_ddf().compute()
