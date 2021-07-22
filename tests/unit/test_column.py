from nvtabular.column import Column, Columns


def test_column():
    column = Column("name", tags=["tag-1"])

    assert column.name == "name"
    assert column.tags[0] == "tag-1"
    assert column.with_name("a").name == "a"
    assert column.with_properties(prop=1).properties["prop"] == 1


def test_columns():
    cols = Columns([Column("a"), [Column("b")]])

    assert all(
        str(col).endswith("suffix") for col in cols.map_names(lambda x: x + "suffix").flatten()
    )
    assert all(col.tags[0] == "tag" for col in cols.map(lambda x: x.with_tags(["tag"])).flatten())
