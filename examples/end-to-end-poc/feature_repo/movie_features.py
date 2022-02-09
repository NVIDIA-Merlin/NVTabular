from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource


movie_features = FileSource(
    path="/home/karl/Projects/nvidia/NVTabular/examples/end-to-end-poc/feature_repo/data/movie_features.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

movie = Entity(name="movieId", value_type=ValueType.INT32, description="movie id",)

movie_features_view = FeatureView(
    name="movie_features",
    entities=["movieId"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="genres", dtype=ValueType.INT64_LIST),
        Feature(name="tags_unique", dtype=ValueType.INT64_LIST),
        Feature(name="tags_nunique", dtype=ValueType.INT32),
    ],
    online=True,
    input=movie_features,
    tags={},
)
