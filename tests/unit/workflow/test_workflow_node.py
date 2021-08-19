import numpy as np
import pytest

from nvtabular import Dataset, Workflow, WorkflowNode, dispatch
from nvtabular.columns import ColumnSelector, Schema
from nvtabular.ops import Categorify, DifferenceLag, FillMissing, Operator, Rename, TargetEncoding
from tests.conftest import assert_eq


def test_selecting_columns_sets_selector_and_kind():
    node = ColumnSelector(["a", "b", "c"]) >> Operator()
    output = node[["a", "b"]]
    assert output.selector.names == ["a", "b"]
    assert "[" in output.kind and "]" in output.kind

    output = node["b"]
    assert output.selector.names == ["b"]
    assert "[" in output.kind and "]" in output.kind


def test_workflow_node_converts_lists_to_selectors():
    node = WorkflowNode([])
    assert node.selector == ColumnSelector([])

    node.selector = ["a", "b", "c"]
    assert node.selector == ColumnSelector(["a", "b", "c"])


def test_input_output_column_names():
    input_node = ["a", "b", "c"] >> FillMissing()
    assert input_node.input_columns.names == ["a", "b", "c"]
    assert input_node.output_columns.names == ["a", "b", "c"]

    chained_node = input_node >> Categorify()
    assert chained_node.input_columns.names == ["a", "b", "c"]
    assert chained_node.output_columns.names == ["a", "b", "c"]

    selection_node = input_node[["b", "c"]]
    assert selection_node.input_columns.names == ["b", "c"]
    assert selection_node.output_columns.names == ["b", "c"]

    addition_node = input_node + ["d"]
    assert addition_node.input_columns.names == ["a", "b", "c", "d"]
    assert addition_node.output_columns.names == ["a", "b", "c", "d"]

    rename_node = input_node >> Rename(postfix="_renamed")
    assert rename_node.input_columns.names == ["a", "b", "c"]
    assert rename_node.output_columns.names == ["a_renamed", "b_renamed", "c_renamed"]

    dependency_node = input_node >> TargetEncoding("d")
    assert dependency_node.input_columns.names == ["a", "b", "c"]
    assert dependency_node.output_columns.names == ["TE_a_d", "TE_b_d", "TE_c_d"]


def test_dependency_column_names():
    dependency_node = ["a", "b", "c"] >> TargetEncoding("d")
    assert dependency_node.dependency_columns.names == ["d"]


def test_workflow_node_addition():
    node1 = ["a", "b"] >> Operator()
    node2 = ["c", "d"] >> Operator()
    node3 = ["e", "f"] >> Operator()

    output_node = node1 + node2
    assert len(output_node.parents) == 2
    assert output_node.output_columns.names == ["a", "b", "c", "d"]

    output_node = node1 + "c"
    assert len(output_node.parents) == 1
    assert output_node.output_columns.names == ["a", "b", "c"]

    output_node = node1 + "c" + "d"
    assert output_node.output_columns.names == ["a", "b", "c", "d"]

    output_node = node1 + node2 + "e"
    assert output_node.output_columns.names == ["a", "b", "c", "d", "e"]

    output_node = node1 + node2 + node3
    assert output_node.output_columns.names == ["a", "b", "c", "d", "e", "f"]

    # Addition with groups
    output_node = node1 + ["c", "d"]
    assert output_node.output_columns.names == ["a", "b", "c", "d"]
    assert output_node.output_columns.grouped_names == ["a", "b", ("c", "d")]

    output_node = node1 + [node2, "e"]
    assert output_node.output_columns.names == ["a", "b", "c", "d", "e"]
    assert output_node.output_columns.grouped_names == ["a", "b", ("c", "d", "e")]

    output_node = node1 + [node2, node3]
    assert output_node.output_columns.names == ["a", "b", "c", "d", "e", "f"]
    assert output_node.output_columns.grouped_names == ["a", "b", ("c", "d", "e", "f")]


def test_workflow_node_dependencies():
    # Full WorkflowNode case
    node1 = ["a", "b"] >> Operator()
    output_node = ["timestamp"] >> DifferenceLag(partition_cols=[node1], shift=[1, -1])
    assert list(output_node.dependencies) == [node1]

    # ColumnSelector case
    output_node = ["timestamp"] >> DifferenceLag(partition_cols=["userid"], shift=[1, -1])
    assert list(output_node.dependencies) == [ColumnSelector(["userid"])]


def test_compute_schemas():
    root_schema = Schema(["a", "b", "c", "d", "e"])

    node1 = ["a", "b"] >> Rename(postfix="_renamed")
    node1.compute_schemas(root_schema)

    assert node1.input_columns.names == ["a", "b"]
    assert node1.output_columns.names == ["a_renamed", "b_renamed"]

    node2 = node1 + "c"
    node2.compute_schemas(root_schema)

    assert node2.input_columns.names == ["a_renamed", "b_renamed", "c"]
    assert node2.output_columns.names == ["a_renamed", "b_renamed", "c"]

    node3 = node2["a_renamed"]
    node3.compute_schemas(root_schema)

    assert node3.input_columns.names == ["a_renamed"]
    assert node3.output_columns.names == ["a_renamed"]


def test_workflow_node_select():
    df = dispatch._make_df({"a": [1, 4, 9, 16, 25], "b": [0, 1, 2, 3, 4], "c": [25, 16, 9, 4, 1]})

    input_features = WorkflowNode(ColumnSelector(["a", "b", "c"]))
    # pylint: disable=unnecessary-lambda
    sqrt_features = input_features[["a", "c"]] >> (lambda col: np.sqrt(col))
    plus_one_features = input_features["b"] >> (lambda col: col + 1)
    features = sqrt_features + plus_one_features

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
