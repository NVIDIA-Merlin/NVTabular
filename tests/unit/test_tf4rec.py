import numpy as np

try:
    from cudf import to_datetime
except ImportError:
    from dask.dataframe import to_datetime

import nvtabular as nvt
from nvtabular import ColumnSelector

NUM_ROWS = 10000


def test_tf4rec():
    inputs = {
        "user_session": np.random.randint(1, 10000, NUM_ROWS),
        "product_id": np.random.randint(1, 51996, NUM_ROWS),
        "category_id": np.random.randint(0, 332, NUM_ROWS),
        "event_time_ts": np.random.randint(1570373000, 1670373390, NUM_ROWS),
        "prod_first_event_time_ts": np.random.randint(1570373000, 1570373382, NUM_ROWS),
        "price": np.random.uniform(0, 2750, NUM_ROWS),
    }
    df = nvt.dispatch._make_df(inputs)

    # categorify features

    cat_feats = (
        ["user_session", "product_id", "category_id"]
        >> nvt.ops.Categorify()
        >> nvt.ops.LambdaOp(lambda col: col + 1)
    )

    # create time features
    sessionTs = ["event_time_ts"]

    sessionTime = (
        sessionTs
        >> nvt.ops.LambdaOp(lambda col: to_datetime(col, unit="s"))
        >> nvt.ops.Rename(name="event_time_dt")
    )

    sessionTime_weekday = (
        sessionTime
        >> nvt.ops.LambdaOp(lambda col: col.dt.weekday)
        >> nvt.ops.Rename(name="et_dayofweek")
    )

    def get_cycled_feature_value_sin(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_sin = np.sin(2 * np.pi * value_scaled)
        return value_sin

    def get_cycled_feature_value_cos(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_cos = np.cos(2 * np.pi * value_scaled)
        return value_cos

    weekday_sin = (
        sessionTime_weekday
        >> (lambda col: get_cycled_feature_value_sin(col + 1, 7))
        >> nvt.ops.Rename(name="et_dayofweek_sin")
    )
    weekday_cos = (
        sessionTime_weekday
        >> (lambda col: get_cycled_feature_value_cos(col + 1, 7))
        >> nvt.ops.Rename(name="et_dayofweek_cos")
    )
    from nvtabular.ops import Operator

    # custom op for item recency
    class ItemRecency(Operator):
        def transform(self, columns, gdf):
            for column in columns.names:
                col = gdf[column]
                item_first_timestamp = gdf["prod_first_event_time_ts"]
                delta_days = (col - item_first_timestamp) / (60 * 60 * 24)
                gdf[column + "_age_days"] = delta_days * (delta_days >= 0)
            return gdf

        def output_column_names(self, columns):
            return ColumnSelector([column + "_age_days" for column in columns.names])

        def dependencies(self):
            return ["prod_first_event_time_ts"]

    recency_features = ["event_time_ts"] >> ItemRecency()
    recency_features_norm = (
        recency_features
        >> nvt.ops.LogOp()
        >> nvt.ops.Normalize()
        >> nvt.ops.Rename(name="product_recency_days_log_norm")
    )

    time_features = (
        sessionTime + sessionTime_weekday + weekday_sin + weekday_cos + recency_features_norm
    )

    # Smoothing price long-tailed distribution
    price_log = (
        ["price"] >> nvt.ops.LogOp() >> nvt.ops.Normalize() >> nvt.ops.Rename(name="price_log_norm")
    )

    # Relative Price to the average price for the category_id
    def relative_price_to_avg_categ(col, gdf):
        epsilon = 1e-5
        col = ((gdf["price"] - col) / (col + epsilon)) * (col > 0).astype(int)
        return col

    avg_category_id_pr = (
        ["category_id"]
        >> nvt.ops.JoinGroupby(cont_cols=["price"], stats=["mean"])
        >> nvt.ops.Rename(name="avg_category_id_price")
    )
    relative_price_to_avg_category = (
        avg_category_id_pr
        >> nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=["price"])
        >> nvt.ops.Rename(name="relative_price_to_avg_categ_id")
    )

    groupby_feats = (
        ["event_time_ts"] + cat_feats + time_features + price_log + relative_price_to_avg_category
    )

    # Define Groupby Workflow
    groupby_features = groupby_feats >> nvt.ops.Groupby(
        groupby_cols=["user_session"],
        sort_cols=["event_time_ts"],
        aggs={
            "product_id": ["list", "count"],
            "category_id": ["list"],
            "event_time_dt": ["first"],
            "et_dayofweek_sin": ["list"],
            "et_dayofweek_cos": ["list"],
            "price_log_norm": ["list"],
            "relative_price_to_avg_categ_id": ["list"],
            "product_recency_days_log_norm": ["list"],
        },
        name_sep="-",
    )

    SESSIONS_MAX_LENGTH = 20
    MINIMUM_SESSION_LENGTH = 2

    groupby_features_nonlist = groupby_features["user_session", "product_id-count"]

    groupby_features_list = groupby_features[
        "price_log_norm-list",
        "product_recency_days_log_norm-list",
        "et_dayofweek_sin-list",
        "et_dayofweek_cos-list",
        "product_id-list",
        "category_id-list",
        "relative_price_to_avg_categ_id-list",
    ]

    groupby_features_trim = (
        groupby_features_list
        >> nvt.ops.ListSlice(0, SESSIONS_MAX_LENGTH)
        >> nvt.ops.Rename(postfix="_seq")
    )

    # calculate session day index based on 'event_time_dt-first' column
    day_index = (
        (groupby_features["event_time_dt-first"])
        >> nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days + 1)
        >> nvt.ops.Rename(f=lambda col: "day_index")
    )

    selected_features = groupby_features_nonlist + groupby_features_trim + day_index

    filtered_sessions = selected_features >> nvt.ops.Filter(
        f=lambda df: df["product_id-count"] >= MINIMUM_SESSION_LENGTH
    )

    dataset = nvt.Dataset(df)

    workflow = nvt.Workflow(filtered_sessions)
    workflow.fit(dataset)
    sessions_gdf = workflow.transform(dataset).to_ddf().compute()

    assert not sessions_gdf.isnull().any().all()
