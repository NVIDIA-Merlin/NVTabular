import numpy as np
import pytest

from nvtabular import Dataset, Workflow, WorkflowNode, dispatch
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.ops import (
    Categorify,
    DifferenceLag,
    FillMissing,
    LambdaOp,
    Operator,
    Rename,
    TargetEncoding,
)
from tests.conftest import assert_eq


def test_selecting_columns_sets_selector_and_kind():
    node = ColumnSelector(["a", "b", "c"]) >> Operator()
    output = node[["a", "b"]]
    assert output.selector.names == ["a", "b"]

    output = node["b"]
    assert output.selector.names == ["b"]


def test_workflow_node_converts_lists_to_selectors():
    node = WorkflowNode([])
    assert node.selector == ColumnSelector([])

    node.selector = ["a", "b", "c"]
    assert node.selector == ColumnSelector(["a", "b", "c"])


def test_input_output_column_names():
    schema = Schema(["a", "b", "c", "d", "e"])

    input_node = ["a", "b", "c"] >> FillMissing()
    workflow = Workflow(input_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["a", "b", "c"]
    assert workflow.output_node.output_columns.names == ["a", "b", "c"]

    chained_node = input_node >> Categorify()
    workflow = Workflow(chained_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["a", "b", "c"]
    assert workflow.output_node.output_columns.names == ["a", "b", "c"]

    selection_node = input_node[["b", "c"]]
    workflow = Workflow(selection_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["b", "c"]
    assert workflow.output_node.output_columns.names == ["b", "c"]

    addition_node = input_node + ["d"]
    workflow = Workflow(addition_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["a", "b", "c", "d"]
    assert workflow.output_node.output_columns.names == ["a", "b", "c", "d"]

    rename_node = input_node >> Rename(postfix="_renamed")
    workflow = Workflow(rename_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["a", "b", "c"]
    assert workflow.output_node.output_columns.names == ["a_renamed", "b_renamed", "c_renamed"]

    dependency_node = input_node >> TargetEncoding("d")
    workflow = Workflow(dependency_node).fit_schema(schema)
    assert workflow.output_node.input_columns.names == ["a", "b", "c"]
    assert workflow.output_node.output_columns.names == ["TE_a_d", "TE_b_d", "TE_c_d"]


def test_dependency_column_names():
    dependency_node = ["a", "b", "c"] >> TargetEncoding("d")
    assert dependency_node.dependency_columns.names == ["d"]


def test_workflow_node_addition():
    schema = Schema(["a", "b", "c", "d", "e", "f"])

    node1 = ["a", "b"] >> Operator()
    node2 = ["c", "d"] >> Operator()
    node3 = ["e", "f"] >> Operator()

    output_node = node1 + node2
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.names == ["a", "b", "c", "d"]

    output_node = node1 + "c"
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.names == ["a", "b", "c"]

    output_node = node1 + "c" + "d"
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.names == ["a", "b", "c", "d"]

    output_node = node1 + node2 + "e"
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.names == ["a", "b", "c", "d", "e"]

    output_node = node1 + node2 + node3
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.names == ["a", "b", "c", "d", "e", "f"]

    # Addition with groups
    output_node = node1 + ["c", "d"]
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.grouped_names == ["a", "b", "c", "d"]

    output_node = node1 + [node2, "e"]
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.grouped_names == ["a", "b", "c", "d", "e"]

    output_node = node1 + [node2, node3]
    workflow = Workflow(output_node).fit_schema(schema)
    assert workflow.output_node.output_columns.grouped_names == ["a", "b", "c", "d", "e", "f"]


def test_workflow_node_subtraction():
    schema = Schema(["a", "b", "c", "d", "e", "f"])

    node1 = ["a", "b", "c", "d"] >> Operator()
    node2 = ["c", "d"] >> Operator()
    node3 = ["b"] >> Operator()

    output_node = node1 - ["c", "d"]
    workflow = Workflow(output_node).fit_schema(schema)
    assert len(output_node.parents) == 1
    assert len(output_node.dependencies) == 0
    assert workflow.output_node.output_columns.names == ["a", "b"]

    output_node = node1 - node2
    workflow = Workflow(output_node).fit_schema(schema)
    assert len(output_node.parents) == 1
    assert len(output_node.dependencies) == 1
    assert workflow.output_node.output_columns.names == ["a", "b"]

    output_node = ["a", "b", "c", "d"] - node2
    workflow = Workflow(output_node).fit_schema(schema)
    assert len(output_node.parents) == 1
    assert len(output_node.dependencies) == 1
    assert workflow.output_node.output_columns.names == ["a", "b"]

    output_node = node1 - ["c", "d"] - node3
    workflow = Workflow(output_node).fit_schema(schema)
    assert len(output_node.parents) == 1
    assert len(output_node.dependencies) == 1
    assert workflow.output_node.output_columns.names == ["a"]


def test_addition_nodes_are_combined():
    schema = Schema(["a", "b", "c", "d", "e", "f", "g", "h"])

    node1 = ["a", "b"] >> Operator()
    node2 = ["c", "d"] >> Operator()
    node3 = ["e", "f"] >> Operator()
    node4 = ["g", "h"] >> Operator()

    add_node = node1 + node2 + node3
    workflow = Workflow(add_node).fit_schema(schema)
    assert set(workflow.output_node.parents) == {node1}
    assert set(workflow.output_node.dependencies) == {node2, node3}
    assert set(workflow.output_node.output_columns.names) == {"a", "b", "c", "d", "e", "f"}

    add_node = node1 + "c" + "d"
    workflow = Workflow(add_node).fit_schema(schema)
    assert set(workflow.output_node.parents) == {node1}
    assert set(workflow.output_node.output_columns.names) == {"a", "b", "c", "d"}

    add_node = "c" + node1 + "d"
    workflow = Workflow(add_node).fit_schema(schema)
    assert set(workflow.output_node.parents) == {node1}
    assert set(workflow.output_node.output_columns.names) == {"a", "b", "c", "d"}

    add_node = node1 + "e" + node2
    workflow = Workflow(add_node).fit_schema(schema)
    assert set(workflow.output_node.parents) == {node1}
    assert node2 in workflow.output_node.dependencies
    assert set(workflow.output_node.output_columns.names) == {"a", "b", "e", "c", "d"}

    add_node1 = node1 + node2
    add_node2 = node3 + node4

    add_node = add_node1 + add_node2
    workflow = Workflow(add_node).fit_schema(schema)

    assert set(workflow.output_node.parents) == {node1}
    assert set(workflow.output_node.dependencies) == {node2, node3, node4}
    assert set(workflow.output_node.output_columns.names) == {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
    }


def test_workflow_node_dependencies():
    # Full WorkflowNode case
    node1 = ["a", "b"] >> Operator()
    output_node = ["timestamp"] >> DifferenceLag(partition_cols=[node1], shift=[1, -1])
    assert list(output_node.dependencies) == [node1]

    # ColumnSelector case
    output_node = ["timestamp"] >> DifferenceLag(partition_cols=["userid"], shift=[1, -1])
    assert output_node.dependencies[0].selector == ColumnSelector(["userid"])


def test_compute_schemas():
    root_schema = Schema(["a", "b", "c", "d", "e"])

    node1 = ["a", "b"] >> Rename(postfix="_renamed")
    node1.parents[0].compute_schemas(root_schema)
    node1.compute_schemas(root_schema)

    assert node1.input_columns.names == ["a", "b"]
    assert node1.output_columns.names == ["a_renamed", "b_renamed"]

    node2 = node1 + "c"
    node2.dependencies[0].compute_schemas(root_schema)
    node2.compute_schemas(root_schema)

    assert node2.input_columns.names == ["a_renamed", "b_renamed", "c"]
    assert node2.output_columns.names == ["a_renamed", "b_renamed", "c"]

    node3 = node2["a_renamed"]
    node3.compute_schemas(root_schema)

    assert node3.input_columns.names == ["a_renamed"]
    assert node3.output_columns.names == ["a_renamed"]


def test_workflow_node_select():
    df = dispatch._make_df({"a": [1, 4, 9, 16, 25], "b": [0, 1, 2, 3, 4], "c": [25, 16, 9, 4, 1]})
    dataset = Dataset(df)

    input_features = WorkflowNode(ColumnSelector(["a", "b", "c"]))
    # pylint: disable=unnecessary-lambda
    sqrt_features = input_features[["a", "c"]] >> (lambda col: np.sqrt(col))
    plus_one_features = input_features["b"] >> (lambda col: col + 1)
    features = sqrt_features + plus_one_features

    workflow = Workflow(features)
    workflow.fit(dataset)

    df_out = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")

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
    dataset = Dataset(df)

    geo_selector = ColumnSelector(["geo"])
    country = (
        geo_selector >> LambdaOp(lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country")
    )
    # country1 = geo_selector >> (lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country1")
    # country2 = geo_selector >> (lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country2")
    user = "user"
    # user2 = "user2"

    # make sure we can do a 'combo' categorify (cross based) of country+user
    # as well as categorifying the country and user columns on their own
    cats = country + user + [country + user] >> Categorify(encode_type="combo")

    workflow = Workflow(cats)
    workflow.fit_schema(dataset.infer_schema())

    df_out = workflow.fit_transform(dataset).to_ddf().compute(scheduler="synchronous")

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
