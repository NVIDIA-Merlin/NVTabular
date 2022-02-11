from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType


user_features = FileSource(
    path="/nvtabular/examples/end-to-end-poc/feature_repo/data/user_features.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

movie = Entity(
    name="user_id",
    value_type=ValueType.INT32,
    description="user id",
)

user_features_view = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="movie_ids", dtype=ValueType.INT64_LIST),
        Feature(name="movie_id_count", dtype=ValueType.INT32),
        Feature(name="search_terms", dtype=ValueType.INT64_LIST),
        Feature(name="genres", dtype=ValueType.INT64_LIST),
    ],
    online=True,
    input=user_features,
    tags={},
)
