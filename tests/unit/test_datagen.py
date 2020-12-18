
import nvtabular as nvt
def test_powerlaw():
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]
    
    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.PowerLawDistro(0.1))
    df_pw = df_gen.create_df(10000, 5, 5, cat_cardinality=[50,50,50,50,50])
    sts, ps = df_gen.verify_df(df_pw[cats])
    assert all(s > 0.9 for s in sts)
    
    
def test_uniform():
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.UniformDistro())
    df_pw = df_gen.create_df(10000, 5, 5, cat_cardinality=[50,50,50,50,50])
    sts, ps = df_gen.verify_df(df_pw[cats])
    assert all(s > 0.9 for s in sts)
