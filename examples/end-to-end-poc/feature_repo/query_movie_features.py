from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    feature_refs=[
        'movie_features:genres',
        'movie_features:tags_unique'
    ],
    entity_rows=[{"movieId": 1000}]
).to_dict()

pprint(feature_vector)
