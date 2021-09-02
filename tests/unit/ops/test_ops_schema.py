import pytest

from nvtabular import ColumnSchema, ColumnSelector, Schema, ops


@pytest.mark.parametrize("properties", [{}, {"p1": "1"}])
@pytest.mark.parametrize("tags", [[], ["TAG1", "TAG2"]])
@pytest.mark.parametrize(
    "op",
    [
        (ops.Bucketize, [1]),
        (ops.Rename, {"postfix": "_trim"}),
        ops.Categorify,
        (ops.Categorify, {"encode_type": "combo"}),
        (ops.Clip, 0),
        (ops.DifferenceLag, "1"),
        ops.FillMissing,
        (ops.Groupby, ["1"]),
        (ops.HashBucket, 1),
        (ops.HashedCross, 1),
        (ops.JoinGroupby, ["1"]),
        (ops.ListSlice, 0),
        ops.LogOp,
        ops.Normalize,
        (ops.TargetEncoding, ["1"]),
    ],
)
@pytest.mark.parametrize("selection", [["1"], ["2", "3"], ["1", "2", "3", "4"]])
def test_schema_out(tags, properties, selection, op):
    # Create columnSchemas
    column_schemas = []
    all_cols = []
    if isinstance(op, tuple):
        if isinstance(op[1], dict):
            op = op[0](**op[1])
        else:
            op = op[0](op[1])
    else:
        op = op()
    for x in range(5):
        all_cols.append(str(x))
        column_schemas.append(ColumnSchema(str(x), tags=tags, properties=properties))
    # Turn to Schema
    schema = Schema(column_schemas)
    # run schema through op
    selector = ColumnSelector(selection)
    new_schema = op.compute_output_schema(schema, selector)
    # check schema otherside
    # should only be selected columns
    if not isinstance(op, ops.HashedCross):
        assert len(new_schema) == len(selection)
    # should have dtype float
    for col_name in selector.names:
        names_group = [name for name in new_schema.column_schemas if col_name in name]
        if names_group:
            for name in names_group:
                schema1 = new_schema.column_schemas[name]
                # should not be exactly the same name, having gone through operator
                assert schema1.dtype == op._get_dtype()
                all_tags = op._get_tags() + tags
                assert len(schema1.tags) == len(all_tags)
                assert set(op._get_tags()).issubset(schema1.tags)
                assert schema1.properties == properties
    not_used = [col for col in all_cols if col not in selector.names]
    for col_name in not_used:
        assert col_name not in new_schema.column_schemas
