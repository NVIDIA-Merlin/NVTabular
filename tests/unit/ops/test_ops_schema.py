import numpy as np
import pytest

import nvtabular as nvt
from merlin.core import dispatch
from merlin.core.dispatch import HAS_GPU
from nvtabular import ColumnSchema, ColumnSelector, Schema, ops


@pytest.mark.parametrize("properties", [{}, {"p1": "1"}])
@pytest.mark.parametrize("tags", [[], ["TAG1", "TAG2"]])
@pytest.mark.parametrize(
    "op",
    [
        ops.Bucketize([1]),
        ops.Rename(postfix="_trim"),
        ops.Categorify(),
        ops.Categorify(encode_type="combo"),
        ops.Clip(0),
        ops.DifferenceLag("1"),
        ops.FillMissing(),
        ops.Groupby(["1"]),
        ops.HashBucket(1),
        ops.HashedCross(1),
        ops.JoinGroupby(["1"]),
        ops.ListSlice(0),
        ops.LogOp(),
        ops.Normalize(),
        ops.TargetEncoding(["1"]),
        ops.AddMetadata(tags=["excellent"], properties={"domain": {"min": 0, "max": 20}}),
        ops.AddTags(tags=["excellent"]),
        ops.AddProperties(properties={"domain": {"min": 0, "max": 20}}),
        ops.TagAsUserID(),
        ops.TagAsItemID(),
        ops.TagAsUserFeatures(),
        ops.TagAsItemFeatures(),
        ops.ValueCount(),
    ],
)
@pytest.mark.parametrize("selection", [["1"], ["2", "3"], ["1", "2", "3", "4"]])
def test_schema_out(tags, properties, selection, op):
    # Create columnSchemas
    column_schemas = []
    all_cols = []
    for x in range(5):
        all_cols.append(str(x))
        column_schemas.append(
            ColumnSchema(str(x), dtype=np.int32, tags=tags, properties=properties)
        )

    # Turn to Schema
    input_schema = Schema(column_schemas)

    # run schema through op
    selector = ColumnSelector(selection)
    output_schema = op.compute_output_schema(input_schema, selector)

    # should have dtype float
    for col_name in selector.names:
        names_group = [name for name in output_schema.column_schemas if col_name in name]
        if names_group:
            for name in names_group:
                result_schema = output_schema.column_schemas[name]

                expected_dtype = op._compute_dtype(
                    ColumnSchema(col_name), Schema([input_schema.column_schemas[col_name]])
                ).dtype

                expected_tags = op._compute_tags(
                    ColumnSchema(col_name), Schema([input_schema.column_schemas[col_name]])
                ).tags

                expected_properties = op._compute_properties(
                    ColumnSchema(col_name), Schema([input_schema.column_schemas[col_name]])
                ).properties

                assert result_schema.dtype == expected_dtype
                if name in selector.names:
                    assert result_schema.properties == expected_properties

                    assert len(result_schema.tags) == len(expected_tags)
                else:
                    assert set(expected_tags).issubset(result_schema.tags)

    not_used = [col for col in all_cols if col not in selector.names]
    for col_name in not_used:
        assert col_name not in output_schema.column_schemas


@pytest.mark.parametrize("properties", [{"p1": "1"}])
@pytest.mark.parametrize("tags", [["TAG1", "TAG2"]])
@pytest.mark.parametrize(
    "op_routine",
    [
        [ops.Categorify()],
        [ops.Clip(min_value=10), ops.Categorify()],
        [ops.Categorify(), ops.Rename(postfix="_test")],
        [ops.Clip(min_value=10), ops.Categorify(), ops.Rename(postfix="_test")],
    ],
)
def test_categorify_schema_properties(properties, tags, op_routine):
    run_op_full(properties, tags, op_routine)


@pytest.mark.parametrize("properties", [{}])
@pytest.mark.parametrize("tags", [[]])
@pytest.mark.parametrize(
    "op_routine",
    [
        [ops.Categorify()],
        [ops.Clip(min_value=10), ops.Categorify()],
        [ops.Categorify(), ops.Rename(postfix="_test")],
        [ops.Clip(min_value=10), ops.Categorify(), ops.Rename(postfix="_test")],
    ],
)
def test_categorify_schema_properties_blank(properties, tags, op_routine):
    run_op_full(properties, tags, op_routine)


@pytest.mark.parametrize("properties", [{}])
@pytest.mark.parametrize("tags", [["TAG1", "TAG2"]])
@pytest.mark.parametrize(
    "op_routine",
    [
        [ops.Categorify()],
        [ops.Clip(min_value=10), ops.Categorify()],
        [ops.Categorify(), ops.Rename(postfix="_test")],
        [ops.Clip(min_value=10), ops.Categorify(), ops.Rename(postfix="_test")],
    ],
)
def test_categorify_schema_properties_tag(properties, tags, op_routine):
    run_op_full(properties, tags, op_routine)


@pytest.mark.parametrize("properties", [{"p1": "1"}])
@pytest.mark.parametrize("tags", [[]])
@pytest.mark.parametrize(
    "op_routine",
    [
        [ops.Categorify()],
        [ops.Clip(min_value=10), ops.Categorify()],
        [ops.Categorify(), ops.Rename(postfix="_test")],
        [ops.Clip(min_value=10), ops.Categorify(), ops.Rename(postfix="_test")],
    ],
)
def test_categorify_schema_properties_props(properties, tags, op_routine):
    run_op_full(properties, tags, op_routine)


def run_op_full(properties, tags, op_routine):
    column_schemas = []
    all_cols = []
    for x in range(5):
        all_cols.append(str(x))
        column_schemas.append(ColumnSchema(str(x), tags=tags, properties=properties))

    # Turn to Schema
    schema = Schema(column_schemas)
    df_dict = {}
    num_rows = 10000
    for column_name in schema.column_names:
        df_dict[column_name] = np.random.randint(1, 1000, num_rows)

    df = dispatch.make_df(df_dict)
    dataset = nvt.Dataset(df)
    test_node = ColumnSelector(schema.column_names) >> op_routine[0]
    for op in op_routine[1:]:
        test_node = test_node >> op
    processor = nvt.Workflow(test_node)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    workflow_schema_out = processor.output_node.output_schema
    for column_name in workflow_schema_out.column_names:
        schema1 = workflow_schema_out.column_schemas[column_name]
        assert "domain" in schema1.properties
        embeddings_info = schema1.properties["domain"]
        # should always exist, represents unknown
        assert embeddings_info["min"] == 0
        assert embeddings_info["max"] == new_gdf[column_name].max() + 1


@pytest.mark.parametrize("properties", [{"p1": "1"}])
@pytest.mark.parametrize("tags", [[]])
@pytest.mark.parametrize(
    "op_routine",
    [
        [ops.Categorify(), ops.Rename(postfix="_test"), ops.ValueCount()],
    ],
)
def test_ops_list_vc(properties, tags, op_routine):
    column_schemas = []
    all_cols = []
    for x in range(5):
        all_cols.append(str(x))
        column_schemas.append(ColumnSchema(str(x), tags=tags, properties=properties))

    # Turn to Schema
    schema = Schema(column_schemas)
    df_dict = {}
    num_rows = 10000
    for column_name in schema.column_names:
        df_dict[column_name] = np.random.randint(1, 1000, num_rows)
        df_dict[column_name] = [[x] * np.random.randint(1, 10) for x in df_dict[column_name]]

    df = dispatch.make_df(df_dict)
    dataset = nvt.Dataset(df)
    test_node = ColumnSelector(schema.column_names) >> op_routine[0]
    for op in op_routine[1:]:
        test_node = test_node >> op
    processor = nvt.Workflow(test_node)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    workflow_schema_out = processor.output_node.output_schema
    for column_name in workflow_schema_out.column_names:
        schema1 = workflow_schema_out.column_schemas[column_name]
        assert "domain" in schema1.properties
        embeddings_info = schema1.properties["domain"]
        # should always exist, represents unknown
        assert embeddings_info["min"] == 0
        if HAS_GPU:
            assert embeddings_info["max"] == new_gdf[column_name]._column.elements.max() + 1
        else:
            list_vals = dispatch.pull_apart_list(new_gdf[column_name])[0]
            assert embeddings_info["max"] == list_vals.max() + 1
        assert "value_count" in schema1.properties
        val_c = schema1.properties["value_count"]
        assert val_c["min"] == op_routine[-1].stats[column_name]["value_count"]["min"]
        assert val_c["max"] == op_routine[-1].stats[column_name]["value_count"]["max"]
