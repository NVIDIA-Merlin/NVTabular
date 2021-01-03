import pytest

import nvtabular as nvt


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_powerlaw(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.PowerLawDistro(0.1))
    df_pw = df_gen.create_df(num_rows, 5, 5, cat_cardinality=[50, 50, 50, 50, 50])
    sts, ps = df_gen.verify_df(df_pw[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_uniform(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, 5, 5, cat_cardinality=[50, 50, 50, 50, 50])
    sts, ps = df_gen.verify_df(df_uni[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_cat_rep(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]
    cats_rep = [(1, 5), (4, 9), (2, 6), (2, 8), (1, 8)]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.UniformDistro())
    df_uni = df_gen.create_df(
        num_rows, 5, 5, cat_cardinality=[50, 50, 50, 50, 50], cats_rep=cats_rep
    )
    df_cats = df_uni.select_dtypes("O")
    assert df_cats.shape[1] == len(cats)
    assert df_cats.shape[0] == num_rows
    for idx, cat in enumerate(cats):
        df_cats[cat].nunique() == 50
        len(df_cats[cat].min()) == cats_rep[idx][0]
        len(df_cats[cat].max()) == cats_rep[idx][1]
