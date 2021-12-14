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
import string

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
from nvtabular import ColumnSelector, dispatch, ops
from tests.conftest import assert_eq

try:
    import dask_cudf

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("cat_groups", ["Author", [["Author", "Engaging-User"]]])
@pytest.mark.parametrize("kfold", [1, 3])
@pytest.mark.parametrize("fold_seed", [None, 42])
@pytest.mark.parametrize("cpu", _CPU)
def test_target_encode(tmpdir, cat_groups, kfold, fold_seed, cpu):
    df = dispatch._make_df(
        {
            "Author": list(string.ascii_uppercase),
            "Engaging-User": list(string.ascii_lowercase),
            "Cost": range(26),
            "Post": [0, 1] * 13,
        }
    )
    if cpu:
        df = dd.from_pandas(df if isinstance(df, pd.DataFrame) else df.to_pandas(), npartitions=3)
    else:
        df = dask_cudf.from_cudf(df, npartitions=3)

    cont_names = ["Cost"]
    te_features = cat_groups >> ops.TargetEncoding(
        cont_names,
        out_path=str(tmpdir),
        kfold=kfold,
        out_dtype="float32",
        fold_seed=fold_seed,
        drop_folds=False,  # Keep folds to validate
    )

    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()
    workflow = nvt.Workflow(te_features + cont_features + ["Author", "Engaging-User"])
    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    df_lib = dispatch.get_lib()
    if kfold > 1:
        # Cat columns are unique.
        # Make sure __fold__ mapping is correct
        if cat_groups == "Author":
            name = "__fold___Author"
            cols = ["__fold__", "Author"]
        else:
            name = "__fold___Author_Engaging-User"
            cols = ["__fold__", "Author", "Engaging-User"]

        check = df_lib.read_parquet(te_features.op.stats[name])
        check = check[cols].sort_values(cols).reset_index(drop=True)
        df_out_check = df_out[cols].sort_values(cols).reset_index(drop=True)
        assert_eq(check, df_out_check, check_dtype=False)


def test_target_encode_group():
    df = dispatch._make_df(
        {
            "Cost": range(15),
            "Post": [1, 2, 3, 4, 5] * 3,
            "Author": ["A"] * 5 + ["B"] * 5 + ["C"] * 2 + ["D"] * 3,
            "Engaging_User": ["A"] * 5 + ["B"] * 3 + ["E"] * 2 + ["D"] * 3 + ["G"] * 2,
        }
    )

    cat_groups = ["Author", "Engaging_User"]
    labels = ColumnSelector(["Post"]) >> ops.LambdaOp(lambda col: (col > 3).astype("int8"))
    te_features = cat_groups >> ops.TargetEncoding(
        labels,
        out_path="./",
        kfold=1,
        out_dtype="float32",
        drop_folds=False,  # Keep folds to validate
    )

    workflow = nvt.Workflow(te_features + ["Author", "Engaging_User"])
    workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")


@pytest.mark.parametrize("npartitions", [1, 2])
@pytest.mark.parametrize("cpu", _CPU)
def test_target_encode_multi(tmpdir, npartitions, cpu):
    cat_1 = np.asarray(["baaaa"] * 12)
    cat_2 = np.asarray(["baaaa"] * 6 + ["bbaaa"] * 3 + ["bcaaa"] * 3)
    num_1 = np.asarray([1, 1, 2, 2, 2, 1, 1, 5, 4, 4, 4, 4])
    num_2 = np.asarray([1, 1, 2, 2, 2, 1, 1, 5, 4, 4, 4, 4]) * 2
    df = dispatch._make_df({"cat": cat_1, "cat2": cat_2, "num": num_1, "num_2": num_2})
    if cpu:
        df = dd.from_pandas(
            df if isinstance(df, pd.DataFrame) else df.to_pandas(), npartitions=npartitions
        )
    else:
        df = dask_cudf.from_cudf(df, npartitions=npartitions)

    cat_groups = ["cat", "cat2", ["cat", "cat2"]]
    te_features = cat_groups >> ops.TargetEncoding(
        ["num", "num_2"], out_path=str(tmpdir), kfold=1, p_smooth=5, out_dtype="float32"
    )

    workflow = nvt.Workflow(te_features)

    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    assert "TE_cat_cat2_num" in df_out.columns
    assert "TE_cat_num" in df_out.columns
    assert "TE_cat2_num" in df_out.columns
    assert "TE_cat_cat2_num_2" in df_out.columns
    assert "TE_cat_num_2" in df_out.columns
    assert "TE_cat2_num_2" in df_out.columns

    assert_eq(df_out["TE_cat2_num"].values, df_out["TE_cat_cat2_num"].values)
    assert_eq(df_out["TE_cat2_num_2"].values, df_out["TE_cat_cat2_num_2"].values)
    assert df_out["TE_cat_num"].iloc[0] != df_out["TE_cat2_num"].iloc[0]
    assert df_out["TE_cat_num_2"].iloc[0] != df_out["TE_cat2_num_2"].iloc[0]
    assert math.isclose(df_out["TE_cat_num"].iloc[0], num_1.mean(), abs_tol=1e-4)
    assert math.isclose(df_out["TE_cat_num_2"].iloc[0], num_2.mean(), abs_tol=1e-3)
