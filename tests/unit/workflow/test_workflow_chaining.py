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


import numpy as np

import nvtabular as nvt
from merlin.core.dispatch import HAS_GPU, make_df
from nvtabular import Dataset, Workflow, ops


def test_chaining_1():
    df = make_df(
        {
            "cont01": np.random.randint(1, 100, 100),
            "cont02": np.random.random(100) * 100,
            "cat01": np.random.randint(0, 10, 100),
            "label": np.random.randint(0, 3, 100),
        }
    )
    df["cont01"][:10] = None

    cont1 = "cont01" >> ops.FillMissing()
    conts = cont1 + "cont02" >> ops.NormalizeMinMax()
    workflow = Workflow(conts + "cat01" + "label")

    result = workflow.fit_transform(Dataset(df)).to_ddf().compute()

    assert result["cont01"].max() <= 1.0
    assert result["cont02"].max() <= 1.0


def test_chaining_2():
    gdf = make_df(
        {
            "A": [1, 2, 2, 9, 6, np.nan, 3],
            "B": [2, np.nan, 4, 7, 7, 2, 5],
            "C": ["a", "b", "c", np.nan, np.nan, "g", "k"],
        }
    )

    cat_names = ["C"]
    cont_names = ["A", "B"]
    label_name = []

    all_features = (
        cat_names + cont_names
        >> ops.LambdaOp(f=lambda col: col.isnull())
        >> ops.Rename(postfix="_isnull")
    )
    cat_features = cat_names >> ops.Categorify()

    workflow = Workflow(all_features + cat_features + label_name)

    dataset = nvt.Dataset(gdf, engine="parquet")

    workflow.fit(dataset)

    result = workflow.transform(dataset).to_ddf().compute()

    assert all(x in list(result.columns) for x in ["A_isnull", "B_isnull", "C_isnull"])
    if HAS_GPU:
        assert (x in result["C"].unique() for x in set(gdf["C"].dropna().to_arrow()))
    else:
        assert (x in result["C"].unique() for x in set(gdf["C"].dropna()))


def test_chaining_3():
    gdf_test = make_df(
        {
            "ad_id": [1, 2, 2, 6, 6, 8, 3, 3],
            "source_id": [2, 4, 4, 7, 5, 2, 5, 2],
            "platform": [1, 2, np.nan, 2, 1, 3, 3, 1],
            "clicked": [1, 0, 1, 0, 0, 1, 1, 0],
        }
    )

    platform_features = ["platform"] >> ops.Dropna()
    joined = ["ad_id"] >> ops.JoinGroupby(cont_cols=["clicked"], stats=["sum", "count"])
    joined_lambda = (
        joined
        >> ops.LambdaOp(f=lambda col, gdf: col / gdf["ad_id_count"])
        >> ops.Rename(postfix="_ctr")
    )

    workflow = Workflow(platform_features + joined + joined_lambda)

    dataset = nvt.Dataset(gdf_test, engine="parquet")

    workflow.fit(dataset)

    result = workflow.transform(dataset).to_ddf().compute()

    assert all(
        x in result.columns for x in ["ad_id_count", "ad_id_clicked_sum_ctr", "ad_id_clicked_sum"]
    )
