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
import os
import random

import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
from merlin.core import dispatch
from merlin.core.compat import HAS_GPU, cudf
from merlin.core.dispatch import make_df
from nvtabular import ColumnSelector, ops
from nvtabular.ops.categorify import get_embedding_sizes
from tests.conftest import assert_eq

if cudf:
    _CPU = [True, False]
else:
    _CPU = [True]
_HAS_GPU = HAS_GPU


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("include_nulls", [True, False])
@pytest.mark.parametrize("cardinality_memory_limit", [None, "24B"])
def test_categorify_size(tmpdir, cpu, include_nulls, cardinality_memory_limit):
    num_rows = 50
    num_distinct = 10

    possible_session_ids = list(range(num_distinct))
    if include_nulls:
        possible_session_ids.append(None)

    df = dispatch.make_df(
        {"session_id": [random.choice(possible_session_ids) for _ in range(num_rows)]},
        device="cpu" if cpu else None,
    )

    cat_features = ["session_id"] >> nvt.ops.Categorify(
        out_path=str(tmpdir),
        cardinality_memory_limit=cardinality_memory_limit,
    )
    workflow = nvt.Workflow(cat_features)
    if cardinality_memory_limit:
        # We set an artificially-low `cardinality_memory_limit`
        # argument to ensure that a UserWarning will be thrown
        with pytest.warns(UserWarning):
            workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute()
    else:
        workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute()

    vals = df["session_id"].value_counts()
    vocab = dispatch.read_dispatch(cpu=cpu)(
        os.path.join(tmpdir, "categories", "unique.session_id.parquet")
    )

    if cpu:
        expected = dict(zip(vals.index, vals))
        computed = {
            session: size
            for session, size in zip(vocab["session_id"], vocab["session_id_size"])
            if size
        }
    else:
        # Ignore first element if it is NaN
        if vocab["session_id"].iloc[:2].isna().any():
            session_id = vocab["session_id"].iloc[2:]
            session_id_size = vocab["session_id_size"].iloc[2:]
        else:
            session_id = vocab["session_id"]
            session_id_size = vocab["session_id_size"]
        expected = dict(zip(vals.index.values_host, vals.values_host))
        computed = {
            session: size
            for session, size in zip(session_id.values_host, session_id_size.values_host)
            if size
        }
    first_key = list(computed.keys())[0]
    if pd.isna(first_key):
        computed.pop(first_key)
    assert computed == expected


def test_na_value_count(tmpdir):
    gdf = dispatch.make_df(
        {
            "productID": ["B00406YHLI"] * 5
            + ["B002YXS8E6"] * 5
            + ["B00011KM38"] * 2
            + [np.nan] * 3,
            "brand": ["Coby"] * 5 + [np.nan] * 5 + ["Cooler Master"] * 2 + ["Asus"] * 3,
        }
    )

    cat_features = ["brand", "productID"] >> nvt.ops.Categorify()
    workflow = nvt.Workflow(cat_features)
    train_dataset = nvt.Dataset(gdf, engine="parquet")
    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_ddf().compute()

    single_meta = dispatch.read_dispatch(fmt="parquet")("./categories/meta.brand.parquet")
    second_meta = dispatch.read_dispatch(fmt="parquet")("./categories/meta.productID.parquet")
    assert single_meta["kind"].iloc[1] == "null"
    assert single_meta["num_observed"].iloc[1] == 5
    assert second_meta["kind"].iloc[1] == "null"
    assert second_meta["num_observed"].iloc[1] == 3


@pytest.mark.parametrize("freq_threshold", [0, 1, 2])
@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("dtype", [None, np.int32, np.int64])
@pytest.mark.parametrize("vocabs", [None, {"Authors": pd.Series([f"User_{x}" for x in "ACBE"])}])
def test_categorify_lists(tmpdir, freq_threshold, cpu, dtype, vocabs):
    df = dispatch.make_df(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors", "Engaging User"]
    label_name = ["Post"]

    cat_features = cat_names >> ops.Categorify(
        out_path=str(tmpdir), freq_threshold=freq_threshold, dtype=dtype, vocabs=vocabs
    )

    workflow = nvt.Workflow(cat_features + label_name)
    df_out = workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute()

    # Columns are encoded independently
    if cpu:
        assert df_out["Authors"][0].dtype == np.dtype(dtype) if dtype else np.dtype("int64")
        compare = [list(row) for row in df_out["Authors"].tolist()]
    else:
        assert df_out["Authors"].dtype == cudf.core.dtypes.ListDtype(dtype if dtype else "int64")
        compare = df_out["Authors"].to_arrow().to_pylist()

    if freq_threshold < 2 or vocabs is not None:
        assert compare == [[3], [3, 6], [5, 4], [4]]
    else:
        assert compare == [[3], [3, 2], [2, 4], [4]]


@pytest.mark.parametrize("cat_names", [[["Author", "Engaging User"]], ["Author", "Engaging User"]])
@pytest.mark.parametrize("kind", ["joint", "combo"])
@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_multi(tmpdir, cat_names, kind, cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    label_name = ["Post"]

    cats = cat_names >> ops.Categorify(out_path=str(tmpdir), encode_type=kind)

    workflow = nvt.Workflow(cats + label_name)

    df_out = (
        workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute(scheduler="synchronous")
    )

    if len(cat_names) == 1:
        if kind == "joint":
            # Columns are encoded jointly
            compare_authors = (
                df_out["Author"].to_list() if cpu else df_out["Author"].to_arrow().to_pylist()
            )
            compare_engaging = (
                df_out["Engaging User"].to_list()
                if cpu
                else df_out["Engaging User"].to_arrow().to_pylist()
            )
            # again userB has highest frequency given lowest encoding
            assert compare_authors == [4, 7, 3, 5]
            assert compare_engaging == [3, 3, 4, 6]
        else:
            # Column combinations are encoded
            compare_engaging = (
                df_out["Author_Engaging User"].to_list()
                if cpu
                else df_out["Author_Engaging User"].to_arrow().to_pylist()
            )
            assert compare_engaging == [3, 6, 4, 5]
    else:
        # Columns are encoded independently
        compare_authors = (
            df_out["Author"].to_list() if cpu else df_out["Author"].to_arrow().to_pylist()
        )
        compare_engaging = (
            df_out["Engaging User"].to_list()
            if cpu
            else df_out["Engaging User"].to_arrow().to_pylist()
        )
        assert compare_authors == [3, 6, 4, 5]
        # User B is first in frequency based ordering
        assert compare_engaging == [3, 3, 4, 5]


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize(
    "cat_names",
    [
        [["Author", "Engaging User"], ["Author"], ["Engaging User"]],
        [["Author", "Engaging User"], ["Author"], "Engaging User"],
        [["Author", "Engaging User"], "Author", ["Engaging User"]],
        [["Author", "Engaging User"], "Author", "Engaging User"],
    ],
)
@pytest.mark.parametrize(
    "input_with_output",
    [
        # dupes in both Author
        {
            "df_data": {
                "Author": ["User_B", "User_E", "User_B", "User_C"],
                "Engaging User": ["User_C", "User_B", "User_A", "User_D"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [3, 5, 3, 4],
            "expected_e": [5, 4, 3, 6],
            "expected_ae": [4, 6, 3, 5],
        },
        # dupes in both Engaging user
        {
            "df_data": {
                "Author": ["User_A", "User_E", "User_B", "User_C"],
                "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [3, 6, 4, 5],
            "expected_e": [3, 3, 4, 5],
            "expected_ae": [3, 6, 4, 5],
        },
        # dupes in both Author and Engaging User
        {
            "df_data": {
                "Author": ["User_C", "User_E", "User_B", "User_C"],
                "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [3, 5, 4, 3],
            "expected_e": [3, 3, 4, 5],
            "expected_ae": [4, 6, 3, 5],
        },
        # dupes in both, lining up
        {
            "df_data": {
                "Author": ["User_A", "User_B", "User_C", "User_C"],
                "Engaging User": ["User_A", "User_B", "User_C", "User_C"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [4, 5, 3, 3],
            "expected_e": [4, 5, 3, 3],
            "expected_ae": [4, 5, 3, 3],
        },
        # no dupes
        {
            "df_data": {
                "Author": ["User_C", "User_E", "User_B", "User_A"],
                "Engaging User": ["User_C", "User_B", "User_A", "User_D"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [5, 6, 4, 3],
            "expected_e": [5, 4, 3, 6],
            "expected_ae": [5, 6, 4, 3],
        },
        # Include null value
        {
            "df_data": {
                "Author": [np.nan, "User_E", "User_B", "User_A"],
                "Engaging User": ["User_C", "User_B", "User_A", "User_D"],
                "Post": [1, 2, 3, 4],
            },
            "expected_a": [1, 5, 4, 3],
            "expected_e": [5, 4, 3, 6],
            "expected_ae": [3, 6, 5, 4],
        },
    ],
)
def test_categorify_multi_combo(tmpdir, input_with_output, cat_names, cpu):
    kind = "combo"
    df = pd.DataFrame(input_with_output["df_data"])

    label_name = ["Post"]
    cats = cat_names >> ops.Categorify(out_path=str(tmpdir), encode_type=kind)
    workflow = nvt.Workflow(cats + label_name)
    df_out = (
        workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute(scheduler="synchronous")
    )

    # Column combinations are encoded
    compare_a = df_out["Author"].to_list() if cpu else df_out["Author"].to_arrow().to_pylist()
    compare_e = (
        df_out["Engaging User"].to_list() if cpu else df_out["Engaging User"].to_arrow().to_pylist()
    )
    compare_ae = (
        df_out["Author_Engaging User"].to_list()
        if cpu
        else df_out["Author_Engaging User"].to_arrow().to_pylist()
    )
    assert compare_a == input_with_output["expected_a"]
    assert compare_e == input_with_output["expected_e"]
    assert compare_ae == input_with_output["expected_ae"]


@pytest.mark.parametrize("freq_limit", [None, 0, {"Author": 3, "Engaging User": 4}])
@pytest.mark.parametrize("buckets", [None, 10, {"Author": 10, "Engaging User": 20}])
@pytest.mark.parametrize("search_sort", [True, False])
@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_freq_limit(tmpdir, freq_limit, buckets, search_sort, cpu):
    if search_sort and cpu:
        # invalid combination - don't test
        return

    df = dispatch.make_df(
        {
            "Author": [
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_B",
                "User_C",
            ],
            "Engaging User": [
                "User_B",
                "User_B",
                "User_A",
                "User_D",
                "User_B",
                "User_c",
                "User_A",
                "User_D",
                "User_D",
                "User_D",
            ],
        }
    )

    isfreqthr = freq_limit > 0 if isinstance(freq_limit, int) else isinstance(freq_limit, dict)

    if (not search_sort and isfreqthr) or (search_sort and not isfreqthr):
        cat_names = ["Author", "Engaging User"]

        cats = cat_names >> ops.Categorify(
            freq_threshold=freq_limit,
            out_path=str(tmpdir),
            search_sorted=search_sort,
            num_buckets=buckets,
        )

        workflow = nvt.Workflow(cats)
        df_out = (
            workflow.fit_transform(nvt.Dataset(df, cpu=cpu))
            .to_ddf()
            .compute(scheduler="synchronous")
        )

        # Check size statistics add up to len(df)
        for col in ["Author", "Engaging User"]:
            check_meta = dispatch.read_dispatch(fmt="parquet")(
                str(tmpdir) + f"/categories/meta.{col}.parquet"
            )
            assert check_meta["num_observed"].sum() == len(df)

        null, oov = 1, 1
        unique = {"Author": 5, "Engaging User": 4}
        freq_limited = {"Author": 2, "Engaging User": 1}
        if freq_limit and not buckets:
            # Column combinations are encoded
            if isinstance(freq_limit, dict):
                assert df_out["Author"].max() == null + oov + freq_limited["Author"]
                assert df_out["Engaging User"].max() == null + oov + freq_limited["Engaging User"]
            else:
                assert len(df["Author"].unique()) == df_out["Author"].max()
                assert len(df["Engaging User"].unique()) == df_out["Engaging User"].max()
        elif not freq_limit and buckets:
            if isinstance(buckets, dict):
                assert df_out["Author"].max() <= null + buckets["Author"] + unique["Author"]
                assert (
                    df_out["Engaging User"].max()
                    <= null + buckets["Engaging User"] + unique["Engaging User"]
                )
            else:
                assert df_out["Author"].max() <= null + buckets + unique["Author"]
                assert df_out["Engaging User"].max() <= null + buckets + unique["Engaging User"]
        elif freq_limit and buckets:
            if (
                isinstance(buckets, dict)
                and isinstance(freq_limit, dict)
                and not isinstance(df, pd.DataFrame)
            ):
                assert df_out["Author"].max() <= null + freq_limited["Author"] + buckets["Author"]
                assert (
                    df_out["Engaging User"].max()
                    <= null + freq_limited["Engaging User"] + buckets["Engaging User"]
                )


@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_hash_bucket_only(cpu):
    df = dispatch.make_df(
        {
            "Authors": ["User_A", "User_A", "User_E", "User_B", "User_C"],
            "Engaging_User": ["User_B", "User_B", "User_A", "User_D", "User_D"],
            "Post": [1, 2, 3, 4, 5],
        }
    )
    cat_names = ["Authors", "Engaging_User"]
    buckets = 10
    max_size = buckets + 2  # Must include pad and null indices
    dataset = nvt.Dataset(df, cpu=cpu)
    hash_features = cat_names >> ops.Categorify(num_buckets=buckets, max_size=max_size)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check hashed values
    assert new_gdf["Authors"].max() <= max_size
    assert new_gdf["Engaging_User"].max() <= max_size
    # check embedding size is equal to the num_buckets after hashing
    assert nvt.ops.get_embedding_sizes(processor)["Authors"][0] == max_size
    assert nvt.ops.get_embedding_sizes(processor)["Engaging_User"][0] == max_size


@pytest.mark.parametrize("max_emb_size", [6, {"Author": 8, "Engaging_User": 7}])
def test_categorify_max_size(max_emb_size):
    df = dispatch.make_df(
        {
            "Author": [
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_D",
                "User_F",
                "User_F",
            ],
            "Engaging_User": [
                "User_B",
                "User_B",
                "User_A",
                "User_D",
                "User_B",
                "User_M",
                "User_A",
                "User_D",
                "User_N",
                "User_F",
                "User_E",
            ],
        }
    )

    cat_names = ["Author", "Engaging_User"]
    buckets = 3
    dataset = nvt.Dataset(df)
    cat_features = cat_names >> ops.Categorify(max_size=max_emb_size, num_buckets=buckets)
    processor = nvt.Workflow(cat_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    if isinstance(max_emb_size, int):
        max_emb_size = {name: max_emb_size for name in cat_names}

    # check encoded values after freq_hashing with fix emb size
    assert new_gdf["Author"].max() <= max_emb_size["Author"] + 1
    assert new_gdf["Engaging_User"].max() <= max_emb_size["Engaging_User"] + 1

    # check embedding size is less than max_size after hashing with fix emb size.
    embedding_sizes = nvt.ops.get_embedding_sizes(processor)
    assert embedding_sizes["Author"][0] <= max_emb_size["Author"] + 1
    assert embedding_sizes["Engaging_User"][0] <= max_emb_size["Engaging_User"] + 1

    # make sure we can also get embedding sizes from the workflow_node
    embedding_sizes = nvt.ops.get_embedding_sizes(cat_features)
    assert embedding_sizes["Author"][0] <= max_emb_size["Author"] + 1
    assert embedding_sizes["Engaging_User"][0] <= max_emb_size["Engaging_User"] + 1


def test_categorify_single_table():
    df = dispatch.make_df(
        {
            "Authors": [None, "User_A", "User_A", "User_E", "User_B", "User_C"],
            "Engaging_User": [None, "User_B", "User_B", "User_A", "User_D", "User_D"],
            "Post": [1, 2, 3, 4, None, 5],
        }
    )
    cat_names = ["Authors", "Engaging_User"]
    dataset = nvt.Dataset(df)
    features = cat_names >> ops.Categorify(single_table=True)
    processor = nvt.Workflow(features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    old_max = 1
    for name in cat_names:
        curr_min = new_gdf[name].min()
        assert old_max <= curr_min
        curr_max = new_gdf[name].max()
        old_max += curr_max


@pytest.mark.parametrize("engine", ["parquet"])
def test_categorify_embedding_sizes(dataset, engine):
    cat_1 = ColumnSelector(["name-cat"]) >> ops.Categorify()
    cat_2 = ColumnSelector(["name-string"]) >> ops.Categorify() >> ops.Rename(postfix="_test")

    workflow = nvt.Workflow(cat_1 + cat_2)
    workflow.fit_transform(dataset)

    assert get_embedding_sizes(workflow) == {"name-cat": (29, 16), "name-string_test": (29, 16)}


def test_categorify_no_nulls():
    # See https://github.com/NVIDIA-Merlin/NVTabular/issues/1325
    df = make_df(
        {
            "user_id": [1, 2, 3, 4, 6, 8, 5, 3] * 10,
            "item_id": [2, 4, 4, 7, 5, 2, 5, 2] * 10,
        },
    )
    workflow = nvt.Workflow(["user_id", "item_id"] >> ops.Categorify())
    workflow.fit(nvt.Dataset(df))

    df = pd.read_parquet("./categories/meta.user_id.parquet")
    assert df["kind"].iloc[1] == "null"
    assert df["num_observed"].iloc[1] == 0


@pytest.mark.parametrize("cat_names", [[["Author", "Engaging User"]], ["Author", "Engaging User"]])
@pytest.mark.parametrize("kind", ["joint", "combo"])
@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_domain_name(tmpdir, cat_names, kind, cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cats = cat_names >> ops.Categorify(out_path=str(tmpdir), encode_type=kind)

    workflow = nvt.Workflow(cats)
    workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute(scheduler="synchronous")

    domain_names = []
    for col_name in workflow.output_schema.column_names:
        domain_names.append(workflow.output_schema[col_name].properties["domain"]["name"])

        assert workflow.output_schema[col_name].properties != {}
        assert "domain" in workflow.output_schema[col_name].properties
        assert "name" in workflow.output_schema[col_name].properties["domain"]

    if len(cat_names) == 1 and kind == "combo":
        # Columns are encoded in combination, so there's only one domain name
        assert len(domain_names) == 1
        assert domain_names[0] == "Author_Engaging User"
    else:
        if len(cat_names) == 1 and kind == "joint":
            # Columns are encoded jointly, so the domain names are the same
            assert len(set(domain_names)) == 1
        else:
            # Columns are encoded independently, so the domain names are different
            assert len(set(domain_names)) > 1


@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_domain_max(cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Post", ["Author", "Engaging User"]]
    cats = cat_names >> nvt.ops.Categorify(encode_type="joint")
    workflow = nvt.Workflow(cats)
    df_transform = workflow.fit_transform(nvt.Dataset(df, cpu=cpu))

    assert df_transform.schema["Post"].properties["domain"]["max"] > 0
    assert df_transform.schema["Author"].properties["domain"]["max"] > 0
    assert df_transform.schema["Engaging User"].properties["domain"]["max"] > 0


def test_categorify_max_size_null_iloc_check():
    gdf = make_df({"C1": [1, np.nan, 3, 4, 3] * 5, "C2": [1, 1, 2, 3, 6] * 5})

    cat_features = ["C1", "C2"] >> nvt.ops.Categorify(max_size=4)

    train_dataset = nvt.Dataset(gdf)

    workflow = nvt.Workflow(cat_features)
    workflow.fit(train_dataset)
    workflow.transform(train_dataset)
    # read back the C1 encoding metadata
    meta_C1 = pd.read_parquet("./categories/meta.C1.parquet")
    assert meta_C1["kind"].iloc[1] == "null"
    assert meta_C1["num_observed"].iloc[1] == 5

    # read back the C2 encoding metadata
    meta_C2 = pd.read_parquet("./categories/meta.C2.parquet")
    assert meta_C2["kind"].iloc[1] == "null"
    assert meta_C2["num_observed"].iloc[1] == 0


@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_joint_list(cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": [
                ["User_B", "User_C"],
                [],
                ["User_A", "User_D"],
                ["User_A"],
            ],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Post", ["Author", "Engaging User"]]
    cats = cat_names >> nvt.ops.Categorify(encode_type="joint")
    workflow = nvt.Workflow(cats)
    df_out = (
        workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute(scheduler="synchronous")
    )

    compare_a = df_out["Author"].to_list() if cpu else df_out["Author"].to_arrow().to_pylist()
    compare_e = (
        df_out["Engaging User"].explode().dropna().to_list()
        if cpu
        else df_out["Engaging User"].explode().dropna().to_arrow().to_pylist()
    )

    assert compare_a == [3, 7, 4, 5]
    assert compare_e == [4, 5, 3, 6, 3]


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("split_out", [2, 3])
@pytest.mark.parametrize("max_size", [0, 6])
@pytest.mark.parametrize("buckets", [None, 3])
def test_categorify_split_out(tmpdir, cpu, split_out, max_size, buckets):
    # Test that the result of split_out>1 is
    # equivalent to that of split_out=1
    df = make_df({"user_id": [1, 2, 3, 4, 6, 8, 5, 3] * 10})
    dataset = nvt.Dataset(df, cpu=cpu)

    kwargs = dict(
        max_size=max_size,
        num_buckets=buckets,
        out_path=str(tmpdir),
    )
    check_path = "/".join([str(tmpdir), "categories/unique.user_id.parquet"])

    workflow_1 = nvt.Workflow(["user_id"] >> ops.Categorify(split_out=1, **kwargs))
    workflow_1.fit(dataset)
    cats_1 = dispatch.read_dispatch(fmt="parquet")(check_path)
    result_1 = workflow_1.transform(dataset).compute()

    workflow_n = nvt.Workflow(["user_id"] >> ops.Categorify(split_out=split_out, **kwargs))
    workflow_n.fit(dataset)
    cats_n = dispatch.read_dispatch(fmt="parquet", collection=True)(check_path).compute(
        scheduler="synchronous"
    )
    result_n = workflow_n.transform(dataset).compute()

    # Make sure categories are the same
    # (Note that pandas may convert int64 to float64,
    # instead of nullable Int64)
    cats_n["user_id"] = cats_n["user_id"].astype(cats_1["user_id"].dtype)
    assert_eq(cats_n, cats_1)

    # Check that transform works
    assert_eq(result_n, result_1)

    # Check for tree_width FutureWarning
    with pytest.warns(FutureWarning):
        nvt.Workflow(["user_id"] >> ops.Categorify(tree_width=8))


def test_categorify_inference():
    num_rows = 100
    a_char, z_char = np.array(["a", "z"]).view("int32")
    input_tensors = {
        "unicode_string": np.random.randint(
            low=a_char, high=z_char, size=num_rows * 10, dtype="int32"
        ).view("U10"),
        "int8_feature": np.random.randint(0, 10, dtype="int8", size=num_rows),
        "int16_feature": np.random.randint(0, 10, dtype="int16", size=num_rows),
        "int32_feature": np.random.randint(0, 10, dtype="int32", size=num_rows),
        "int64_feature": np.random.randint(0, 10, dtype="int64", size=num_rows),
        "uint8_feature": np.random.randint(0, 10, dtype="uint8", size=num_rows),
        "uint16_feature": np.random.randint(0, 10, dtype="uint16", size=num_rows),
        "uint32_feature": np.random.randint(0, 10, dtype="uint32", size=num_rows),
        "uint64_feature": np.random.randint(0, 10, dtype="uint64", size=num_rows),
    }
    df = dispatch.make_df(input_tensors)
    cat_names = df.columns
    cats = cat_names >> nvt.ops.Categorify()
    workflow = nvt.Workflow(cats)
    workflow.fit(nvt.Dataset(df))
    model_config = {}
    inference_op = cats.op.inference_initialize(cats.input_columns, model_config)
    output_tensors = inference_op.transform(cats.input_columns, input_tensors)
    for key in input_tensors:
        assert output_tensors[key].dtype == np.dtype("int64")


def test_categorify_transform_only_nans_column():
    train_df = make_df({"cat_column": ["a", "a", "b", "c", np.nan]})
    cat_features = ["cat_column"] >> nvt.ops.Categorify()
    train_dataset = nvt.Dataset(train_df)

    workflow = nvt.Workflow(cat_features)
    workflow.fit(train_dataset)

    inference_df = make_df({"cat_column": [np.nan] * 10})
    inference_dataset = nvt.Dataset(inference_df)

    output = workflow.transform(inference_dataset).compute()
    assert len(output) == len(inference_df)
