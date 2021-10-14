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
from nvtabular.dispatch import HAS_GPU

if HAS_GPU:
    _CPU = [True, False]
    _HAS_GPU = True
else:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
@pytest.mark.parametrize("add_binary_cols", [True, False])
@pytest.mark.parametrize("cpu", _CPU)
def test_fill_median(
    tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, add_binary_cols, cpu
):
    cont_features = op_columns >> nvt.ops.FillMedian(add_binary_cols=add_binary_cols)
    processor = nvt.Workflow(cont_features)

    ds = nvt.Dataset(dataset.to_ddf(), cpu=cpu)
    df0 = df
    if cpu and not isinstance(df0, pd.DataFrame):
        df0 = df0.to_pandas()

    processor.fit(ds)
    new_df = processor.transform(ds).to_ddf().compute()
    new_df.index = df0.index  # Make sure index is aligned for checks
    for col in op_columns:
        col_median = df[col].dropna().quantile(0.5, interpolation="linear")
        assert math.isclose(col_median, processor.output_node.op.medians[col], rel_tol=1e1)
        assert np.all((df0[col].fillna(col_median) - new_df[col]).abs().values <= 1e-2)
        assert (f"{col}_filled" in new_df.keys()) == add_binary_cols
        if add_binary_cols:
            assert df0[col].isna().sum() == new_df[f"{col}_filled"].sum()


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("add_binary_cols", [True, False])
@pytest.mark.parametrize("cpu", _CPU)
def test_fill_missing(tmpdir, df, engine, add_binary_cols, cpu):
    if cpu and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    cont_names = ["x", "y"]
    cont_features = cont_names >> nvt.ops.FillMissing(fill_val=42, add_binary_cols=add_binary_cols)

    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    df = df.reset_index()
    dataset = nvt.Dataset(df, cpu=cpu)
    processor = nvt.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    for col in cont_names:
        assert np.all((df[col].fillna(42) - new_gdf[col]).abs().values <= 1e-2)
        assert new_gdf[col].isna().sum() == 0
        assert (f"{col}_filled" in new_gdf.keys()) == add_binary_cols
        if add_binary_cols:
            assert df[col].isna().sum() == new_gdf[f"{col}_filled"].sum()
