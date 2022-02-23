from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType


movie_features = FileSource(
    path="/nvtabular/examples/end-to-end-poc/feature_repo/data/movie_features.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

movie = Entity(
    name="movie_id",
    value_type=ValueType.INT32,
    description="movie id",
)

movie_features_view = FeatureView(
    name="movie_features",
    entities=["movie_id"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="genres", dtype=ValueType.INT32_LIST),
        Feature(name="tags_unique", dtype=ValueType.INT32_LIST),
        Feature(name="tags_nunique", dtype=ValueType.INT32),
    ],
    online=True,
    input=movie_features,
    tags={},
)
