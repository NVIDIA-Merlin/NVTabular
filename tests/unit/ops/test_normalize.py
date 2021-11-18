#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
import nvtabular.io
from nvtabular import ColumnSelector, dispatch, ops
from nvtabular.dispatch import HAS_GPU, _flatten_list_column, _flatten_list_column_values
from tests.conftest import assert_eq

if HAS_GPU:
    _CPU = [True, False]
    _HAS_GPU = True
else:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
# TODO: dask workflow doesn't support min/max on string columns, so won't work
# with op_columns=None
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
@pytest.mark.parametrize("cpu", _CPU)
def test_normalize_minmax(tmpdir, dataset, gpu_memory_frac, engine, op_columns, cpu):
    df = dataset.to_ddf().compute()
    cont_features = op_columns >> ops.NormalizeMinMax()
    processor = nvtabular.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    for col in op_columns:
        col_min = df[col].min()
        assert col_min == pytest.approx(processor.output_node.op.mins[col], 1e-2)
        col_max = df[col].max()
        assert col_max == pytest.approx(processor.output_node.op.maxs[col], 1e-2)
        df[col] = (df[col] - processor.output_node.op.mins[col]) / (
            processor.output_node.op.maxs[col] - processor.output_node.op.mins[col]
        )
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
def test_normalize(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_features = op_columns >> ops.Normalize()
    processor = nvtabular.Workflow(cont_features)
    processor.fit(dataset)

    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    for col in op_columns:
        assert math.isclose(df[col].mean(), processor.output_node.op.means[col], rel_tol=1e-4)
        assert math.isclose(df[col].std(), processor.output_node.op.stds[col], rel_tol=1e-4)
        df[col] = (df[col] - processor.output_node.op.means[col]) / processor.output_node.op.stds[
            col
        ]
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)

    # our normalize op also works on dicts of cupy/numpy tensors. make sure this works like we'd
    # expect
    df = dataset.compute()
    cupy_inputs = {col: df[col].values for col in op_columns}
    cupy_outputs = cont_features.op.transform(ColumnSelector(op_columns), cupy_inputs)
    for col in op_columns:
        assert np.allclose(cupy_outputs[col], new_gdf[col].values)


@pytest.mark.parametrize("cpu", _CPU)
def test_normalize_lists(tmpdir, cpu):
    df = dispatch._make_df(device="cpu" if cpu else "gpu")
    df["vals"] = [
        [0.0, 1.0, 2.0],
        [
            3.0,
            4.0,
        ],
        [5.0],
    ]

    features = ["vals"] >> nvt.ops.Normalize()
    workflow = nvt.Workflow(features)
    transformed = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    expected = _flatten_list_column_values(df["vals"]).astype("float32")
    expected = (expected - expected.mean()) / expected.std()
    expected_df = type(transformed)({"vals": expected})

    assert_eq(expected_df, _flatten_list_column(transformed["vals"]))


@pytest.mark.parametrize("cpu", _CPU)
def test_normalize_std_zero(cpu):
    df = pd.DataFrame({"a": 7 * [10]})
    dataset = nvt.Dataset(df, cpu=cpu)
    processor = nvtabular.Workflow(["a"] >> ops.Normalize())
    processor.fit(dataset)
    result = processor.transform(dataset).compute()["a"]
    assert (result == 0).all()


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("op_columns", [["x"]])
def test_normalize_upcastfloat64(tmpdir, dataset, gpu_memory_frac, engine, op_columns):
    df = dispatch._make_df({"x": [1.9e10, 2.3e16, 3.4e18, 1.6e19], "label": [1.0, 0.0, 1.0, 0.0]})

    cont_features = op_columns >> ops.Normalize()
    processor = nvtabular.Workflow(cont_features)
    dataset = nvt.Dataset(df)
    processor.fit(dataset)

    new_gdf = processor.transform(dataset).to_ddf().compute()

    for col in op_columns:
        assert math.isclose(df[col].mean(), processor.output_node.op.means[col], rel_tol=1e-4)
        assert math.isclose(df[col].std(), processor.output_node.op.stds[col], rel_tol=1e-4)
        df[col] = (df[col] - processor.output_node.op.means[col]) / processor.output_node.op.stds[
            col
        ]
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)
