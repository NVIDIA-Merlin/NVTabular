import numpy as np
import pytest

import nvtabular as nvt
from nvtabular import ColumnSchema, ColumnSelector, Schema, dispatch, ops
from nvtabular.dispatch import HAS_GPU


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
        column_schemas.append(ColumnSchema(str(x), tags=tags, properties=properties))

    # Turn to Schema
    schema = Schema(column_schemas)

    # run schema through op
    selector = ColumnSelector(selection)
    new_schema = op.compute_output_schema(schema, selector)

    # should have dtype float
    for col_name in selector.names:
        names_group = [name for name in new_schema.column_schemas if col_name in name]
        if names_group:
            for name in names_group:
                schema1 = new_schema.column_schemas[name]

                # should not be exactly the same name, having gone through operator
                assert schema1.dtype == op.output_dtype()
                if name in selector.names:
                    assert (
                        schema1.properties
                        == op._add_properties(schema.column_schemas[schema1.name]).properties
                    )
                    all_tags = op.output_tags() + tags
                    assert len(schema1.tags) == len(all_tags)
                else:
                    assert set(op.output_tags()).issubset(schema1.tags)

    not_used = [col for col in all_cols if col not in selector.names]
    for col_name in not_used:
        assert col_name not in new_schema.column_schemas


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

    df = dispatch._make_df(df_dict)
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

    df = dispatch._make_df(df_dict)
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
            list_vals = nvt.dispatch._pull_apart_list(new_gdf[column_name])[0]
            assert embeddings_info["max"] == list_vals.max() + 1
        assert "value_count" in schema1.properties
        val_c = schema1.properties["value_count"]
        assert val_c["min"] == op_routine[-1].stats[column_name]["value_count"]["min"]
        assert val_c["max"] == op_routine[-1].stats[column_name]["value_count"]["max"]
