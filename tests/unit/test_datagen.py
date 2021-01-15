import cudf
import numpy as np
import pytest

import nvtabular.tools.data_gen as datagen

json_sample = {
    "conts": {
        "cont_1": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_2": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_3": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_4": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_5": {"dtype": np.float32, "min_val": 0, "max_val": 1},
    },
    "cats": {
        "cat_1": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
            "multi_min": 2,
            "multi_max": 5,
            "multi_avg": 3,
        },
        "cat_2": {"dtype": None, "cardinality": 50, "min_entry_size": 1, "max_entry_size": 5},
        "cat_3": {"dtype": None, "cardinality": 50, "min_entry_size": 1, "max_entry_size": 5},
        "cat_4": {"dtype": None, "cardinality": 50, "min_entry_size": 1, "max_entry_size": 5},
        "cat_5": {"dtype": None, "cardinality": 50, "min_entry_size": 1, "max_entry_size": 5},
    },
    "labs": {"lab_1": {"dtype": None, "cardinality": 2}},
}

distros = {
    "cont_1": {"name": "powerlaw", "params": {"alpha": 0.1}},
    "cont_2": {"name": "powerlaw", "params": {"alpha": 0.2}},
    "cat_1": {"name": "powerlaw", "params": {"alpha": 0.1}},
    "cat_2": {"name": "powerlaw", "params": {"alpha": 0.2}},
}


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("distro", [None, distros])
def test_powerlaw(num_rows, distro):
    cats = list(json_sample["cats"].keys())[1:]

    cols = datagen._get_cols_from_schema(json_sample, distros=distro)

    df_gen = datagen.DatasetGen(datagen.PowerLawDistro(0.1))
    df_pw = cudf.DataFrame()
    for x in range(10):
        df_pw_1 = df_gen.create_df(num_rows, cols)
        df_pw = cudf.concat([df_pw, df_pw_1], axis=0)
    sts, ps = df_gen.verify_df(df_pw[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("distro", [None, distros])
def test_uniform(num_rows, distro):
    cats = list(json_sample["cats"].keys())[1:]
    cols = datagen._get_cols_from_schema(json_sample, distros=distro)

    df_gen = datagen.DatasetGen(datagen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, cols)
    sts, ps = df_gen.verify_df(df_uni[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("distro", [None, distros])
def test_width(num_rows, distro):
    json_sample_1 = {
        "conts": {
            "cont_1": {"dtype": np.float32, "min_val": 0, "max_val": 1, "width": 20},
        }
    }
    cols = datagen._get_cols_from_schema(json_sample_1, distros=distro)

    df_gen = datagen.DatasetGen(datagen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, cols)
    assert df_uni.shape[1] == 20


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("distro", [None, distros])
def test_cat_rep(num_rows, distro):
    cats = list(json_sample["cats"].keys())
    cols = datagen._get_cols_from_schema(json_sample, distros=distro)

    df_gen = datagen.DatasetGen(datagen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, cols, entries=True)
    df_cats = df_uni[cats]
    assert df_cats.shape[1] == len(cats)
    assert df_cats.shape[0] == num_rows
    cats_rep = cols["cats"]
    for idx, cat in enumerate(cats[1:]):
        assert df_uni[cat].nunique() == cats_rep[idx + 1].cardinality
        assert df_uni[cat].str.len().min() == cats_rep[idx + 1].min_entry_size
        assert df_uni[cat].str.len().max() == cats_rep[idx + 1].max_entry_size
    check_ser = cudf.Series(df_uni[cats[0]]._column.elements.values_host)
    assert check_ser.nunique() == cats_rep[0].cardinality
    assert check_ser.str.len().min() == cats_rep[0].min_entry_size
    assert check_ser.str.len().max() == cats_rep[0].max_entry_size


def test_json_convert():
    cols = datagen._get_cols_from_schema(json_sample)
    assert len(cols["conts"]) == len(json_sample["conts"].keys())
    assert len(cols["cats"]) == len(json_sample["cats"].keys())
    assert len(cols["labs"]) == len(json_sample["labs"].keys())


@pytest.mark.parametrize("num_rows", [1000, 100000])
@pytest.mark.parametrize("distro", [None, distros])
def test_full_df(num_rows, tmpdir, distro):
    cats = list(json_sample["cats"].keys())
    cols = datagen._get_cols_from_schema(json_sample, distros=distro)

    df_gen = datagen.DatasetGen(datagen.UniformDistro(), gpu_frac=0.00001)
    df_files = df_gen.full_df_create(num_rows, cols, entries=True, output=tmpdir)
    test_size = 0
    full_df = cudf.DataFrame()
    for fi in df_files:
        df = cudf.read_parquet(fi)
        test_size = test_size + df.shape[0]
        full_df = cudf.concat([full_df, df])
    assert test_size == num_rows
    conts_rep = cols["conts"]
    cats_rep = cols["cats"]
    labs_rep = cols["labs"]
    assert df.shape[1] == len(conts_rep) + len(cats_rep) + len(labs_rep)
    for idx, cat in enumerate(cats[1:]):
        dist = cats_rep[idx + 1].distro or df_gen.dist
        if type(full_df[cat]._column) is not cudf.core.column.string.StringColumn:
            sts, ps = dist.verify(full_df[cat].to_pandas())
            assert all(s > 0.9 for s in sts)
        assert full_df[cat].nunique() == cats_rep[idx + 1].cardinality
        assert full_df[cat].str.len().min() == cats_rep[idx + 1].min_entry_size
        assert full_df[cat].str.len().max() == cats_rep[idx + 1].max_entry_size
    check_ser = cudf.Series(full_df[cats[0]]._column.elements.values_host)
    dist = cats_rep[0].distro or df_gen.dist
    if type(full_df[cat]._column) is not cudf.core.column.string.StringColumn:
        sts, ps = dist.verify(full_df[cats[0].to_pandas()])
        assert all(s > 0.9 for s in sts)
    assert check_ser.nunique() == cats_rep[0].cardinality
    assert check_ser.str.len().min() == cats_rep[0].min_entry_size
    assert check_ser.str.len().max() == cats_rep[0].max_entry_size
