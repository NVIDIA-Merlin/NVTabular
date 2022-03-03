# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings

import pandas as pd
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

import nvtabular as nvt
from merlin.dag import ColumnSelector
from nvtabular.ops import Bucketize, Categorify, HashBucket, HashedCross, Rename


def _make_categorical_embedding(name, vocab_size, embedding_dim):
    column = tf.feature_column.categorical_column_with_identity(name, vocab_size)
    if embedding_dim is None:
        return tf.feature_column.indicator_column(column)
    else:
        return tf.feature_column.embedding_column(column, embedding_dim)


def make_feature_column_workflow(feature_columns, label_name, category_dir=None):
    """
    Maps a list of TensorFlow `feature_column`s to an NVTabular `Workflow` which
    imitates their preprocessing functionality. Returns both the finalized
    `Workflow` as well as a list of `feature_column`s that can be used to
    instantiate a `layers.ScalarDenseFeatures` layer to map from `Workflow`
    outputs to dense network inputs. Useful for replacing feature column
    online preprocessing with NVTabular GPU-accelerated online preprocessing
    for faster training.

    Parameters
    ----------
    feature_columns: list(tf.feature_column)
        List of TensorFlow feature columns to emulate preprocessing functions
        of. Doesn't support sequence columns.
    label_name: str
        Name of label column in dataset
    category_dir: str or None
        Directory in which to save categories from vocabulary list and
        vocabulary file columns. If left as None, will create directory
        `/tmp/categories` and save there

    Returns
    -------
    workflow: nvtabular.Workflow
        An NVTabular `Workflow` which performs the preprocessing steps
        defined in `feature_columns`
    new_feature_columns: list(feature_columns)
        List of TensorFlow feature columns that correspond to the output
        from `workflow`. Only contains numeric and identity categorical columns.
    """
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

    _CATEGORIFY_COLUMNS = (fc.VocabularyListCategoricalColumn, fc.VocabularyFileCategoricalColumn)
    categorifies, hashes, crosses, buckets, replaced_buckets = {}, {}, {}, {}, {}

    numeric_columns = []
    new_feature_columns = []
    for column in feature_columns:
        # TODO: check for shared embedding or weighted embedding columns?
        # Do they just inherit from EmbeddingColumn?
        if not isinstance(column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            if isinstance(column, (fc.BucketizedColumn)):
                # bucketized column being fed directly to model means it's
                # implicitly wrapped into an indicator column
                cat_column = column
                embedding_dim = None
            else:
                # can this be anything else? I don't think so
                assert isinstance(column, fc.NumericColumn)

                # check to see if we've seen a bucketized column
                # that gets fed by this feature. If we have, note
                # that it shouldn't be replaced
                if column.key in replaced_buckets:
                    buckets[column.key] = replaced_buckets.pop(column.key)

                numeric_columns.append(column)
                continue
        else:
            cat_column = column.categorical_column

            # use this to keep track of what should be embedding
            # and what should be indicator, makes the bucketized
            # checking easier
            if isinstance(column, fc.EmbeddingColumn):
                embedding_dim = column.dimension
            else:
                embedding_dim = None

        if isinstance(cat_column, fc.BucketizedColumn):
            key = cat_column.source_column.key

            # check if the source numeric column is being fed
            # directly to the model. Keep track of both the
            # boundaries and embedding dim so that we can wrap
            # with either indicator or embedding later
            if key in [col.key for col in numeric_columns]:
                buckets[key] = (column.boundaries, embedding_dim)
            else:
                replaced_buckets[key] = (column.boundaries, embedding_dim)

            # put off dealing with these until the end so that
            # we know whether we need to replace numeric
            # columns or create a separate feature column
            # for them
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
            categorifies[cat_column.key] = list(vocab)
            key = cat_column.key

        elif isinstance(cat_column, fc.HashedCategoricalColumn):
            hashes[cat_column.key] = cat_column.hash_bucket_size
            key = cat_column.key

        elif isinstance(cat_column, fc.CrossedColumn):
            keys = []
            for key in cat_column.keys:
                if isinstance(key, fc.BucketizedColumn):
                    keys.append(key.source_column.key + "_bucketized")
                elif isinstance(key, str):
                    keys.append(key)
                else:
                    keys.append(key.key)
            crosses[tuple(keys)] = (cat_column.hash_bucket_size, embedding_dim)

            # put off making the new columns here too so that we
            # make sure we have the key right after we check
            # for buckets later
            continue

        elif isinstance(cat_column, fc.IdentityCategoricalColumn):
            new_feature_columns.append(column)
            continue

        else:
            raise ValueError("Unknown column {}".format(cat_column))

        new_feature_columns.append(
            _make_categorical_embedding(key, cat_column.num_buckets, embedding_dim)
        )

    features = ColumnSelector(label_name)

    if len(buckets) > 0:
        new_buckets = {}
        for key, (boundaries, embedding_dim) in buckets.items():
            new_feature_columns.append(
                _make_categorical_embedding(key + "_bucketized", len(boundaries) + 1, embedding_dim)
            )
            new_buckets[key] = boundaries

        features_buckets = (
            new_buckets.keys() >> Bucketize(new_buckets) >> Rename(postfix="_bucketized")
        )
        features += features_buckets

    if len(replaced_buckets) > 0:
        new_replaced_buckets = {}
        for key, (boundaries, embedding_dim) in replaced_buckets.items():
            new_feature_columns.append(
                _make_categorical_embedding(key, len(boundaries) + 1, embedding_dim)
            )
            new_replaced_buckets[key] = boundaries
        features_replaced_buckets = new_replaced_buckets.keys() >> Bucketize(new_replaced_buckets)
        features += features_replaced_buckets

    if len(categorifies) > 0:
        vocabs = {column: pd.Series(vocab) for column, vocab in categorifies.items()}
        features += ColumnSelector(list(categorifies.keys())) >> Categorify(vocabs=vocabs)

    if len(hashes) > 0:
        features += ColumnSelector(list(hashes.keys())) >> HashBucket(hashes)

    if len(crosses) > 0:
        # need to check if any bucketized columns are coming from
        # the bucketized version or the raw version
        new_crosses = {}
        for keys, (hash_bucket_size, embedding_dim) in crosses.items():
            # if we're bucketizing the input we have to do more work here -
            if any(key.endswith("_bucketized") for key in keys):
                cross_columns = []
                for key in keys:
                    if key.endswith("_bucketized"):
                        bucketized_cols = []
                        bucketized_cols.append(key)
                        key = key.replace("_bucketized", "")
                        if key in buckets:
                            # find if there are different columns
                            diff_col = list(set(features_buckets.columns) ^ set(bucketized_cols))
                            if diff_col:
                                features_buckets.columns.remove(diff_col[0])
                            cross_columns.append(features_buckets)
                        elif key in replaced_buckets:
                            diff_col = list(
                                set(features_replaced_buckets.columns) ^ set(bucketized_cols)
                            )
                            if diff_col:
                                features_replaced_buckets.columns.remove(diff_col[0])
                            cross_columns.append(features_replaced_buckets)
                        else:
                            raise RuntimeError(f"Unknown bucket column {key}")
                    else:
                        cross_columns.append(nvt.WorkflowNode(key))

                features += sum(cross_columns[1:], cross_columns[0]) >> HashedCross(
                    hash_bucket_size
                )

            else:
                new_crosses[tuple(keys)] = hash_bucket_size
            key = "_X_".join(keys)
            new_feature_columns.append(
                _make_categorical_embedding(key, hash_bucket_size, embedding_dim)
            )

        if new_crosses:
            features += new_crosses.keys() >> HashedCross(new_crosses)

    if numeric_columns:
        features += [col.key for col in numeric_columns]

    workflow = nvt.Workflow(features)

    return workflow, numeric_columns + new_feature_columns
