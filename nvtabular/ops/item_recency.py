from nvtabular.ops import Operator

FIRST_SEEN_ITEM_COL_NAME = "item_ts_first"


class ItemRecency(Operator):
    FIRST_SEEN_ITEM_COL_NAME = FIRST_SEEN_ITEM_COL_NAME

    def __init__(
        self, first_seen_column_name=FIRST_SEEN_ITEM_COL_NAME, out_col_suffix="/age_days"
    ) -> None:
        super().__init__()
        self.first_seen_column_name = first_seen_column_name
        self.out_col_suffix = out_col_suffix

    @classmethod
    def add_first_seen_col_to_df(
        cls, df, item_id_column="item_id", first_seen_column_name=FIRST_SEEN_ITEM_COL_NAME
    ):
        items_first_ts_df = (
            df.groupby(item_id_column)
            .agg({"timestamp": "min"})
            .reset_index()
            .rename(columns={"timestamp": first_seen_column_name})
        )
        merged_df = df.merge(items_first_ts_df, on=["item_id"], how="left")

        return merged_df

    def transform(self, columns, gdf):
        for column in columns:
            col = gdf[column]
            item_first_timestamp = gdf[self.first_seen_column_name]
            delta_days = (col - item_first_timestamp).dt.days
            gdf[column + self.out_col_suffix] = delta_days * (delta_days >= 0)
        return gdf

    def output_column_names(self, columns):
        return [column + self.out_col_suffix for column in columns]

    def dependencies(self):
        return [self.first_seen_column_name]
