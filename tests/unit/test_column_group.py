import cudf
import pytest
from cudf.tests.utils import assert_eq

from nvtabular import ColumnGroup, Dataset, Workflow
from nvtabular.ops import Categorify, Rename


def test_column_group_select():
    df = cudf.DataFrame({"a": [1, 4, 9, 16, 25], "b": [0, 1, 2, 3, 4], "c": [25, 16, 9, 4, 1]})

    input_features = ColumnGroup(["a", "b", "c"])
    sqrt_features = input_features[["a", "c"]] >> cudf.sqrt
    plus_one_features = input_features["b"] >> (lambda col: col + 1)
    features = sqrt_features + plus_one_features

    workflow = Workflow(features)
    df_out = workflow.fit_transform(Dataset(df)).to_ddf().compute(scheduler="synchronous")

    expected = cudf.DataFrame()
    expected["a"] = cudf.sqrt(df["a"])
    expected["c"] = cudf.sqrt(df["c"])
    expected["b"] = df["b"] + 1

    assert_eq(expected, df_out)


def test_nested_column_group():
    df = cudf.DataFrame(
        {
            "geo": ["US>CA", "US>NY", "CA>BC", "CA>ON"],
            "user": ["User_A", "User_A", "User_A", "User_B"],
        }
    )

    country = (
        ColumnGroup(["geo"]) >> (lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country")
    )

    # make sure we can do a 'combo' categorify (cross based) of country+user
    # as well as categorifying the country and user columns on their own
    cats = [country + "user"] + country + "user" >> Categorify(encode_type="combo")

    workflow = Workflow(cats)
    df_out = workflow.fit_transform(Dataset(df)).to_ddf().compute(scheduler="synchronous")

    geo_country = df_out["geo_country"]
    assert geo_country[0] == geo_country[1]  # rows 0,1 are both 'US'
    assert geo_country[2] == geo_country[3]  # rows 2,3 are both 'CA'

    user = df_out["user"]
    assert user[0] == user[1] == user[2]
    assert user[3] != user[2]

    geo_country_user = df_out["geo_country_user"]
    assert geo_country_user[0] == geo_country_user[1]  # US / userA
    assert geo_country_user[2] != geo_country_user[0]  # same user but in canada

    # make sure we get an exception if we nest too deeply (can't handle arbitrarily deep
    # nested column groups - and the exceptions we would get in operators like Categorify
    # are super confusing for users)
    with pytest.raises(ValueError):
        cats = [[country + "user"] + country + "user"] >> Categorify(encode_type="combo")
