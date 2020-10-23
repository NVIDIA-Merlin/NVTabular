import os
import warnings
import yaml

import cudf
import nvtabular as nvt
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc


def make_feature_column_workflow(feature_columns, label_name, category_dir=None):
    # TODO: should we support a dict input for feature columns
    # for multi-tower support?

    def _get_parents(column):
        """
        quick utility function for getting all the input tensors
        that will feed into a column
        """
        # column has no parents, so we've reached a terminal node
        if isinstance(column, str) or isinstance(column.parents[0], str):
            return [column]

        # else climb family tree
        parents = []
        for parent in column.parents:
            parents.extend([i for i in _get_parents(parent) if i not in parents])
        return parents

    # could be more effiient with sets but this is deterministic which
    # might be helpful? Still not sure about this so being safe
    base_columns = []
    for column in feature_columns:
        parents = _get_parents(column)
        base_columns.extend([col for col in parents if col not in base_columns])

    cat_names, cont_names = [], []
    for column in base_columns:
        if isinstance(column, str):
            # cross column input
            # TODO: this means we only accept categorical inputs to
            # cross? How do we generalize this? Probably speaks to
            # the inefficiencies of feature columns as a schema
            # representation
            cat_names.extend(column)
        elif isinstance(column, fc.CategoricalColumn):
            cat_names.extend(column.key)
        else:
            cont_names.extend(column.key)
    workflow = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=[label_name])

    _CATEGORIFY_COLUMNS = (fc.VocabularyListCategoricalColumn, fc.VocabularyFileCategoricalColumn)
    categorifies, hashes, crosses, buckets = {}, {}, {}, {}
    new_feature_columns = []
    for column in feature_columns:
        if not isinstance(column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            # bucketized column being fed directly to model
            if isinstance(column, (fc.BucketizedColumn)):
                cat_column = column
                embedding_dim = None
            else:
                new_feature_columns.append(column)
                continue
        else:
            cat_column = column.categorical_column
            if isinstance(column, fc.EmbeddingColumn):
                embedding_dim = column.dimension
            else:
                embedding_dim = -1

        if isinstance(cat_column, fc.BucketizedColumn):
            # TODO: how do we handle case where both original
            # and bucketized column get fed to model?
            key = cat_column.source_column.key
            buckets[key] = column.boundaries

            if embedding_dim is None:
                # bucketized values being fed as numeric
                # probably a rare case, but worth covering here
                new_feature_columns.append(
                    tf.feature_column.numeric_column(key, cat_column.source_column.shape)
                )
                continue

        elif isinstance(cat_column, _CATEGORIFY_COLUMNS):
            if cat_column.num_oov_buckets > 1:
                warnings.warn("More than 1 oov bucket not supported for Categorify")

            if isinstance(cat_column, _CATEGORIFY_COLUMNS[1]):
                # TODO: how do we handle the case where it's too big to load?
                with open(cat_column.vocab_file, "r") as f:
                    vocab = f.read().split("\n")
            else:
                vocab = cat_column.vocabulary_list
            categorifies[cat_column.key] = vocab

        elif isinstance(cat_column, fc.HashedCategoricalColumn):
            hashes[cat_column.key] = cat_column.hash_bucket_size

        elif isinstance(cat_column, fc.CrossedColumn):
            keys = []
            for key in cat_column.keys:
                if isinstance(key, fc.BucketizedColumn):
                    keys.append(key.source_column.key)
                elif isinstance(key, str):
                    keys.append(key)
                else:
                    keys.append(key.key)
            crosses[tuple(keys)] = cat_column.hash_bucket_size

        new_cat_col = tf.feature_column.categorical_column_with_identity(
            cat_column.key, cat_column.num_buckets
        )
        if embedding_dim < 0:
            new_feature_columns.append(tf.feature_column.indicator_column(new_cat_col))
        else:
            new_feature_columns.append(
                tf.feature_column.embedding_column(new_cat_col, embedding_dim)
            )

    workflow.add_cont_preprocess(nvt.ops.Bucketize(buckets, replace=True))
    workflow.add_cat_preprocess(
        [
            nvt.ops.Categorify(columns=[key for key in categorifies.keys()]),
            nvt.ops.HashBucket(hashes),
        ]
    )
    workflow.add_cat_feature(nvt.ops.HashedCross(crosses))
    workflow.finalize()

    if category_dir is None:
        category_dir = "/tmp/categories"
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    stats = {"categories": {}}
    for feature_name, categories in categorifies.items():
        categories.insert(0, None)
        df = cudf.DataFrame({feature_name: categories})

        save_path = os.path.join(category_dir, f"unique.{feature_name}.parquet")
        df.to_parquet(save_path)
        stats["categories"][feature_name] = save_path

    stats_path = os.path.join(category_dir, "stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(f, stats)
    workflow.load_stats(stats_path)
    return workflow, new_feature_columns
