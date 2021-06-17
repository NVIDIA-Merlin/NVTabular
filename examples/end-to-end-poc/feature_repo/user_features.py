from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource


user_features = FileSource(
    path="/home/karl/Projects/nvidia/NVTabular/examples/end-to-end-poc/feature_repo/data/user_features.parquet",
    event_timestamp_column="datetime",
    created_timestamp_column="created",
)

movie = Entity(name="userId", value_type=ValueType.INT32, description="user id",)

user_features_view = FeatureView(
    name="user_features",
    entities=["userId"],
    ttl=Duration(seconds=86400 * 7),
    features=[
        Feature(name="movieId", dtype=ValueType.INT64_LIST),
        Feature(name="movieId_count", dtype=ValueType.INT32),
        Feature(name="sampled_tag", dtype=ValueType.INT64_LIST),
        Feature(name="genre", dtype=ValueType.INT64_LIST),
    ],
    online=True,
    input=user_features,
    tags={},
)
