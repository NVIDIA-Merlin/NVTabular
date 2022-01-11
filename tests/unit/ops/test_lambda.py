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
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq as assert_eq_dd
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
import nvtabular.io
from nvtabular import ColumnSelector, ops

try:
    import cupy as cp

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cpu", _CPU)
def test_lambdaop(tmpdir, df, paths, gpu_memory_frac, engine, cpu):
    dataset = nvt.Dataset(paths, cpu=cpu)
    df_copy = df.copy()

    # Substring
    # Replacement
    substring = ColumnSelector(["name-cat", "name-string"]) >> ops.LambdaOp(
        lambda col: col.str.slice(1, 3)
    )
    processor = nvtabular.Workflow(substring)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"].str.slice(1, 3), check_index=False)
    assert_eq_dd(new_gdf["name-string"], df_copy["name-string"].str.slice(1, 3), check_index=False)

    # No Replacement from old API (skipped for other examples)
    substring = (
        ColumnSelector(["name-cat", "name-string"])
        >> ops.LambdaOp(lambda col: col.str.slice(1, 3))
        >> ops.Rename(postfix="_slice")
    )
    processor = nvtabular.Workflow(substring + ["name-cat", "name-string"])
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(
        new_gdf["name-cat_slice"],
        df_copy["name-cat"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(
        new_gdf["name-string_slice"],
        df_copy["name-string"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"], check_index=False)
    assert_eq_dd(new_gdf["name-string"], df_copy["name-string"], check_index=False)

    # Replace
    # Replacement
    oplambda = ColumnSelector(["name-cat", "name-string"]) >> ops.LambdaOp(
        lambda col: col.str.replace("e", "XX")
    )
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"].str.replace("e", "XX"), check_index=False)
    assert_eq_dd(
        new_gdf["name-string"], df_copy["name-string"].str.replace("e", "XX"), check_index=False
    )

    # astype
    # Replacement
    oplambda = ColumnSelector(["id"]) >> ops.LambdaOp(lambda col: col.astype(float))
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert new_gdf["id"].dtype == "float64"

    # Workflow
    # Replacement
    oplambda = (
        ColumnSelector(["name-cat"])
        >> ops.LambdaOp(lambda col: col.astype(str).str.slice(0, 1))
        >> ops.Categorify()
    )
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert is_integer_dtype(new_gdf["name-cat"].dtype)

    oplambda = (
        ColumnSelector(["name-cat", "name-string"]) >> ops.Categorify() >> (lambda col: col + 100)
    )
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert is_integer_dtype(new_gdf["name-cat"].dtype)
    assert np.sum(new_gdf["name-cat"] < 100) == 0


@pytest.mark.parametrize("cpu", _CPU)
def test_lambdaop_misalign(cpu):
    size = 12
    df0 = pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.random.choice(["apple", "banana", "orange"], size),
            "c": np.random.choice([0, 1], size),
        }
    )

    ddf0 = dd.from_pandas(df0, npartitions=4)

    cont_names = ColumnSelector(["a"])
    cat_names = ColumnSelector(["b"])
    label = ColumnSelector(["c"])
    if cpu:
        label_feature = label >> ops.LambdaOp(lambda col: np.where(col == 4, 0, 1))
    else:
        label_feature = label >> ops.LambdaOp(lambda col: cp.where(col == 4, 0, 1))
    workflow = nvt.Workflow(cat_names + cont_names + label_feature)

    dataset = nvt.Dataset(ddf0, cpu=cpu)
    transformed = workflow.transform(dataset)
    assert_eq_dd(
        df0[["a", "b"]],
        transformed.to_ddf().compute()[["a", "b"]],
        check_index=False,
    )
