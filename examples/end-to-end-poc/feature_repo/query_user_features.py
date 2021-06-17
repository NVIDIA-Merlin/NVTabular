from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    feature_refs=[
        'user_features:genre',
    ],
    entity_rows=[{"userId": 1000}]
).to_dict()

print(feature_vector)
