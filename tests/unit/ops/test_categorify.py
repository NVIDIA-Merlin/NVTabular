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
from nvtabular import ColumnSelector, dispatch, ops
from nvtabular.ops.categorify import get_embedding_sizes

try:
    import cudf

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("include_nulls", [True, False])
@pytest.mark.parametrize("cardinality_memory_limit", [None, "24B"])
def test_categorify_size(tmpdir, cpu, include_nulls, cardinality_memory_limit):
    num_rows = 50
    num_distinct = 10

    possible_session_ids = list(range(num_distinct))
    if include_nulls:
        possible_session_ids.append(None)

    df = dispatch._make_df(
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
    vocab = dispatch._read_dispatch(cpu=cpu)(
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
        if vocab["session_id"].iloc[:1].isna().any():
            session_id = vocab["session_id"].iloc[1:]
            session_id_size = vocab["session_id_size"].iloc[1:]
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
    gdf = dispatch._make_df(
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

    single_cat = dispatch._read_dispatch("./categories/unique.brand.parquet")(
        "./categories/unique.brand.parquet"
    )
    second_cat = dispatch._read_dispatch("./categories/unique.productID.parquet")(
        "./categories/unique.productID.parquet"
    )
    assert single_cat["brand_size"][0] == 5
    assert second_cat["productID_size"][0] == 3


@pytest.mark.parametrize("freq_threshold", [0, 1, 2])
@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("dtype", [None, np.int32, np.int64])
@pytest.mark.parametrize("vocabs", [None, {"Authors": pd.Series([f"User_{x}" for x in "ACBE"])}])
def test_categorify_lists(tmpdir, freq_threshold, cpu, dtype, vocabs):
    df = dispatch._make_df(
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
        assert compare == [[1], [1, 4], [3, 2], [2]]
    else:
        assert compare == [[1], [1, 0], [0, 2], [2]]


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("start_index", [1, 2, 16])
def test_categorify_lists_with_start_index(tmpdir, cpu, start_index):
    df = dispatch._make_df(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors", "Engaging User"]
    label_name = ["Post"]
    dataset = nvt.Dataset(df, cpu=cpu)
    cat_features = cat_names >> ops.Categorify(out_path=str(tmpdir), start_index=start_index)
    processor = nvt.Workflow(cat_features + label_name)
    processor.fit(dataset)
    df_out = processor.transform(dataset).to_ddf().compute()

    if cpu:
        compare = [list(row) for row in df_out["Authors"].tolist()]
    else:
        compare = df_out["Authors"].to_arrow().to_pylist()

    # Note that start_index is the start_index of the range of encoding, which
    # includes both an initial value for the encoding for out-of-vocabulary items,
    # as well as the values for the rest of the in-vocabulary items.
    # In this group of tests below, there are no out-of-vocabulary items, so our start index
    # value does not appear in the expected comparison object.
    if start_index == 0:
        assert compare == [[1], [1, 4], [3, 2], [2]]
    elif start_index == 1:
        assert compare == [[2], [2, 5], [4, 3], [3]]
    elif start_index == 16:
        assert compare == [[17], [17, 20], [19, 18], [18]]

    # We expect five entries in the embedding size, one for each author,
    # plus start_index many additional entries for our offset start_index.
    embeddings = nvt.ops.get_embedding_sizes(processor)

    assert embeddings[1]["Authors"][0] == (5 + start_index)


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
            assert compare_authors == [2, 5, 1, 3]
            assert compare_engaging == [1, 1, 2, 4]
        else:
            # Column combinations are encoded
            compare_engaging = (
                df_out["Author_Engaging User"].to_list()
                if cpu
                else df_out["Author_Engaging User"].to_arrow().to_pylist()
            )
            assert compare_engaging == [1, 4, 2, 3]
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
        assert compare_authors == [1, 4, 2, 3]
        # User B is first in frequency based ordering
        assert compare_engaging == [1, 1, 2, 3]


@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_multi_combo(tmpdir, cpu):
    cat_names = [["Author", "Engaging User"], ["Author"], "Engaging User"]
    kind = "combo"
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
    assert compare_a == [1, 4, 2, 3]
    # here User B has more frequency so lower encode value
    assert compare_e == [1, 1, 2, 3]
    assert compare_ae == [1, 4, 2, 3]


@pytest.mark.parametrize("freq_limit", [None, 0, {"Author": 3, "Engaging User": 4}])
@pytest.mark.parametrize("buckets", [None, 10, {"Author": 10, "Engaging User": 20}])
@pytest.mark.parametrize("search_sort", [True, False])
@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_freq_limit(tmpdir, freq_limit, buckets, search_sort, cpu):
    if search_sort and cpu:
        # invalid combination - don't test
        return

    df = dispatch._make_df(
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

        if freq_limit and not buckets:
            # Column combinations are encoded
            if isinstance(freq_limit, dict):
                assert df_out["Author"].max() == 2
                assert df_out["Engaging User"].max() == 1
            else:
                assert len(df["Author"].unique()) == df_out["Author"].max()
                assert len(df["Engaging User"].unique()) == df_out["Engaging User"].max()
        elif not freq_limit and buckets:
            if isinstance(buckets, dict):
                assert df_out["Author"].max() <= 9
                assert df_out["Engaging User"].max() <= 19
            else:
                assert df_out["Author"].max() <= 9
                assert df_out["Engaging User"].max() <= 9
        elif freq_limit and buckets:
            if (
                isinstance(buckets, dict)
                and isinstance(buckets, dict)
                and not isinstance(df, pd.DataFrame)
            ):
                assert (
                    df_out["Author"].max()
                    <= (df["Author"].hash_values() % buckets["Author"]).max() + 2 + 1
                )
                assert (
                    df_out["Engaging User"].max()
                    <= (df["Engaging User"].hash_values() % buckets["Engaging User"]).max() + 1 + 1
                )


@pytest.mark.parametrize("cpu", _CPU)
def test_categorify_hash_bucket(cpu):
    df = dispatch._make_df(
        {
            "Authors": ["User_A", "User_A", "User_E", "User_B", "User_C"],
            "Engaging_User": ["User_B", "User_B", "User_A", "User_D", "User_D"],
            "Post": [1, 2, 3, 4, 5],
        }
    )
    cat_names = ["Authors", "Engaging_User"]
    buckets = 10
    dataset = nvt.Dataset(df, cpu=cpu)
    hash_features = cat_names >> ops.Categorify(num_buckets=buckets)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check hashed values
    assert new_gdf["Authors"].max() <= (buckets - 1)
    assert new_gdf["Engaging_User"].max() <= (buckets - 1)
    # check embedding size is equal to the num_buckets after hashing
    assert nvt.ops.get_embedding_sizes(processor)["Authors"][0] == buckets
    assert nvt.ops.get_embedding_sizes(processor)["Engaging_User"][0] == buckets


@pytest.mark.parametrize("max_emb_size", [6, {"Author": 8, "Engaging_User": 7}])
def test_categorify_max_size(max_emb_size):
    df = dispatch._make_df(
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
    assert new_gdf["Author"].max() <= max_emb_size["Author"]
    assert new_gdf["Engaging_User"].max() <= max_emb_size["Engaging_User"]

    # check embedding size is less than max_size after hashing with fix emb size.
    embedding_sizes = nvt.ops.get_embedding_sizes(processor)
    assert embedding_sizes["Author"][0] <= max_emb_size["Author"]
    assert embedding_sizes["Engaging_User"][0] <= max_emb_size["Engaging_User"]

    # make sure we can also get embedding sizes from the workflow_node
    embedding_sizes = nvt.ops.get_embedding_sizes(cat_features)
    assert embedding_sizes["Author"][0] <= max_emb_size["Author"]
    assert embedding_sizes["Engaging_User"][0] <= max_emb_size["Engaging_User"]


def test_categorify_single_table():
    df = dispatch._make_df(
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

    old_max = 0
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

    assert get_embedding_sizes(workflow) == {"name-cat": (27, 16), "name-string_test": (27, 16)}
