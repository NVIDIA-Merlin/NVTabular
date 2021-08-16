import numpy as np
import pytest

from nvtabular import Dataset, Workflow, WorkflowNode, dispatch
from nvtabular.columns import ColumnSelector
from nvtabular.ops import Categorify, Operator, Rename
from tests.conftest import assert_eq


def test_selecting_columns_sets_selector_and_kind():
    node = ColumnSelector(["a", "b", "c"]) >> Operator()
    output = node[["a", "b"]]
    assert output.selector.names == ["a", "b"]
    assert "[" in output.kind and "]" in output.kind

    output = node["b"]
    assert output.selector.names == ["b"]
    assert "[" in output.kind and "]" in output.kind


def test_workflow_node_rshift_doesnt_set_selector():
    node = ColumnSelector(["a", "b", "c"]) >> Operator() >> Operator()
    assert node.selector is None


def test_adding_workflow_nodes_doesnt_set_selector():
    node1 = ColumnSelector(["a", "b", "c"]) >> Operator()
    node2 = ColumnSelector(["d", "e", "f"]) >> Operator()
    output = node1 + node2
    assert output.selector is None


def test_adding_column_names_to_node():
    # TODO: Try out addition nodes that don't create extra
    # WorkflowNodes to be their parents and just add additional
    # columns to their own selectors
    node = ColumnSelector(["a", "b", "c"]) >> Operator()
    output_node = node + "d"

    assert len(output_node.parents) == 1
    assert output_node.selector.names == ["a", "b", "c", "d"]

    node = ColumnSelector(["a", "b", "c"]) >> Operator()
    output_node = node + ["d", "e", "f"]

    assert len(output_node.parents) == 1
    assert len(output_node.parents[0].parents) == 0
    assert output_node.selector.names == ["a", "b", "c", "d", "e", "f"]


def test_workflow_node_select():
    df = dispatch._make_df({"a": [1, 4, 9, 16, 25], "b": [0, 1, 2, 3, 4], "c": [25, 16, 9, 4, 1]})

    input_features = ColumnSelector(["a", "b", "c"])
    # pylint: disable=unnecessary-lambda
    sqrt_features = input_features >> (lambda col: np.sqrt(col))
    plus_one_features = input_features >> (lambda col: col + 1)
    features = sqrt_features[["a", "c"]] + plus_one_features["b"]

    workflow = Workflow(features)
    df_out = workflow.fit_transform(Dataset(df)).to_ddf().compute(scheduler="synchronous")

    expected = dispatch._make_df()
    expected["a"] = np.sqrt(df["a"])
    expected["c"] = np.sqrt(df["c"])
    expected["b"] = df["b"] + 1

    assert_eq(expected, df_out)


def test_nested_workflow_node():
    df = dispatch._make_df(
        {
            "geo": ["US>CA", "US>NY", "CA>BC", "CA>ON"],
            "user": ["User_A", "User_A", "User_A", "User_B"],
        }
    )

    geo_selector = ColumnSelector(["geo"])
    country = geo_selector >> (lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country")

    # make sure we can do a 'combo' categorify (cross based) of country+user
    # as well as categorifying the country and user columns on their own
    cats = country + "user" + [country + "user"] >> Categorify(encode_type="combo")

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


def test_workflow_node_converts_lists():
    node = WorkflowNode([])
    assert node.selector == ColumnSelector([])

    node.selector = ["a", "b", "c"]
    assert node.selector == ColumnSelector(["a", "b", "c"])
