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

import gc

import pandas as pd
import tensorflow as tf
from pandas_tfrecords.from_tfrecords import _get_feature_type, read_example
from tqdm import tqdm


def convert_tfrecords_to_parquet(
    filenames, output_dir, compression_type="", chunks=100000, convert_lists=False
):
    """
    Converts tfrecord files to parquet file format

    Parameters
    ----------
    filenames: list
        List of tfrecord filenames, which should be converted
    output_dir: str
        Output path where the parquet files will be stored
    compression_type: str
        Compression type of the tfrecords. Options: `""` (no compression), `"ZLIB"`, or `"GZIP"`
    chunks: int
        Split tfrecords into multiple parquet files
    convert_lists: Boolean
        Output of tfrecords are lists. Set True to convert lists with fixed length to
        individual columns in the output dataframe

    """
    # TODO: provide row_groupby_size parameter for parquet files
    # TODO: optimize parquet files

    for file in filenames:
        dataset = tf.data.TFRecordDataset(file, compression_type=compression_type)
        features = _detect_schema(dataset)
        parser = read_example(features)
        parsed = dataset.map(parser)
        _to_parquet(parsed, file, output_dir, chunks, convert_lists)


def _detect_schema(dataset):
    # inspired by
    # https://github.com/schipiga/pandas-tfrecords/blob/master/pandas_tfrecords/from_tfrecords.py
    features = {}

    serialized = next(iter(dataset.map(lambda serialized: serialized)))
    seq_ex = tf.train.SequenceExample.FromString(serialized.numpy())

    if seq_ex.context.feature:
        for key, feature in seq_ex.context.feature.items():
            features[key] = tf.io.FixedLenSequenceFeature(
                (), _get_feature_type(feature=feature), allow_missing=True
            )

    return features


def _to_parquet(tfrecords, file, output_dir, chunks, convert_lists):
    out = []
    i = 0
    j = 0
    for tfrecord in tqdm(tfrecords):
        row = {key: val.numpy() for key, val in tfrecord.items()}
        out.append(row)
        i += 1
        if i == chunks:
            df = pd.DataFrame(out)
            if convert_lists:
                df = _convert_lists(df)
            df.to_parquet(
                output_dir + file.split("/")[-1].split(".")[0] + "_" + str(j) + ".parquet"
            )
            i = 0
            out = []
            j += 1
            del df
            gc.collect()
    if len(out) > 0:
        df = pd.DataFrame(out)
        if convert_lists:
            df = _convert_lists(df)
        df.to_parquet(output_dir + file.split("/")[-1].split(".")[0] + "_" + str(j) + ".parquet")
        del df


def _convert_lists(df):
    for col in df.columns:
        series_length = df[col].apply(lambda x: len(x))
        if series_length.var() == 0 and series_length.min() > 0:
            if series_length.max() == 1:
                df[col] = df[col].apply(lambda x: x[0])
            else:
                for i in range(series_length.max()):
                    df[col + "_" + str(i)] = df[col].apply(lambda x: x[i])
    return df
