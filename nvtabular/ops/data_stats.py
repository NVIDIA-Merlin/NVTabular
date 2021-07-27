#
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
import logging
import os

import dask.array as da
import dask.dataframe as dd
import numpy as np
from tensorflow_metadata.proto.v0 import path_pb2, statistics_pb2

from ..column import Columns
from ..dispatch import annotate
from ..visualization import statistics as vis_statistics
from .data_schema import prepare_schema
from .stat_operator import StatOperator

LOG = logging.getLogger("nvtabular")


class DataStats(StatOperator):
    STATS_FILE_NAME = vis_statistics.STATS_FILE_NAME

    def __init__(self, output_path=None, cross_columns=None, name="data"):
        super().__init__()  # pylint: disable=bad-super-call
        self.cross_columns = cross_columns
        self.name = name
        self.stats_path = os.path.join(output_path, self.STATS_FILE_NAME) if output_path else None
        self.output_path = output_path
        self.col_names = []
        self.col_types = []
        self.col_dtypes = []
        self.col_is_lists = []
        self.stats = None

    @classmethod
    def calculate_on_dataset(
        cls, dataset, column_group=None, cross_columns=None, output_path=None, client=None
    ):
        from nvtabular.workflow import Workflow

        column_group = column_group or list(dataset.to_ddf().columns)
        stats = cls(output_path=output_path, cross_columns=cross_columns)

        workflow = Workflow(column_group >> stats, client=client)
        workflow.fit(dataset)

        return stats

    @classmethod
    def load(cls, directory):
        stats_file = os.path.join(directory, cls.STATS_FILE_NAME)

        if not os.path.exists(stats_file):
            return None

        LOG.info("Loading stats from %s", stats_file)
        d = statistics_pb2.DatasetFeatureStatisticsList()
        with open(stats_file, "rb") as f:
            d.ParseFromString(f.read())

        return vis_statistics.DatasetCollectionStatistics(d)

    @annotate("Statistics_fit", color="green", domain="nvt_python")
    def fit(self, columns: Columns, ddf: dd.DataFrame):
        dask_stats = {}

        ddf_dtypes = ddf.head(1)
        num_examples = len(ddf)

        # TODO: Fix median, right now it gives an error.
        # dask_stats["median"] = ddf[columns].quantile(q=0.5, method="dask")

        # For each column, calculate the stats
        for col in columns:
            dask_stats[col] = {}
            self.col_names.append(col)
            # Get dtype for all
            dtype = str(ddf_dtypes[col].dtype)
            is_list = False

            domain = ddf[col]

            if dtype == "list":
                domain = ddf[col].map_partitions(lambda x: x.list.leaves, meta=("x", int))
                lengths = ddf[col].map_partitions(lambda x: x.list.len(), meta=("x", int))
                dask_stats[col]["min_length"] = lengths.min()
                dask_stats[col]["max_length"] = lengths.max()
                dask_stats[col]["mean_length"] = lengths.mean()
                dtype = ddf_dtypes[col].dtype.leaf_type
                is_list = True

            self.col_dtypes.append(dtype)
            self.col_is_lists.append(is_list)

            dask_stats[col]["num_missing"] = len(domain[domain.isnull()])
            dask_stats[col]["num_non_missing"] = domain.count()

            # Get cardinality for cats
            if dtype == "object":
                dask_stats[col]["nunique"] = domain.nunique()
                str_len = domain.map_partitions(lambda x: x.str.len(), meta=("x", int))
                dask_stats[col]["avg_str_length"] = str_len.mean()
                dask_stats[col]["top_values"] = domain.value_counts().nlargest(n=50)

            elif dtype in ["int8", "int32", "int64", "float32", "float64"]:
                # Get various stats
                dask_stats[col]["min"] = domain.min()
                dask_stats[col]["max"] = domain.max()
                dask_stats[col]["mean"] = domain.mean()
                dask_stats[col]["std"] = domain.std()

                dask_stats[col]["num_zeroes"] = (domain == 0).sum()

                h, bins = da.histogram(
                    domain.to_dask_array(), 10, range=[domain.min(), domain.max()]
                )
                dask_stats[col]["histogram"] = h
                dask_stats[col]["histogram_bins"] = bins
            else:
                LOG.warning("%s has unknown type: %s", col, dtype)

            if self.cross_columns:
                dask_stats["corr"] = ddf[self.cross_columns].corr()
                dask_stats["cov"] = ddf[self.cross_columns].cov()

        return num_examples, dask_stats

    def fit_finalize(self, stats):
        num_examples, dask_stats = stats

        self.stats = statistics_pb2.DatasetFeatureStatistics(
            name=self.name, num_examples=num_examples
        )

        for i, col in enumerate(self.col_names):
            dtype = self.col_dtypes[i]
            is_list = self.col_is_lists[i]

            feature = self.stats.features.add()
            feature.name = col

            common_stats = statistics_pb2.CommonStatistics(
                num_non_missing=int(dask_stats[col]["num_non_missing"]),
                num_missing=int(dask_stats[col]["num_missing"]),
                min_num_values=int(dask_stats[col]["min_length"]) if is_list else 1,
                max_num_values=int(dask_stats[col]["max_length"]) if is_list else 1,
                avg_num_values=float(dask_stats[col]["mean_length"]) if is_list else 1.0,
            )

            if dtype in ["int8", "int32", "int64", "float32", "float64"]:
                hist = statistics_pb2.Histogram(type=0)
                h, bins = dask_stats[col]["histogram"], dask_stats[col]["histogram_bins"]
                for i, _ in enumerate(h):
                    bucket = hist.buckets.add()
                    bucket.low_value = bins[i]
                    bucket.high_value = bins[i + 1]
                    bucket.sample_count = float(h[i])

                feature.num_stats.CopyFrom(
                    statistics_pb2.NumericStatistics(
                        min=dask_stats[col]["min"].item(),
                        max=dask_stats[col]["max"].item(),
                        histograms=[hist],
                        common_stats=common_stats,
                        #                     median=1.0,
                        #                     median=float(dask_stats[col]["median"]),
                        mean=dask_stats[col]["mean"].item(),
                        std_dev=dask_stats[col]["std"].item(),
                        num_zeros=dask_stats[col]["num_zeroes"].item(),
                    )
                )
                feature.type = 1 if np.issubdtype(dtype, np.floating) else 0
            elif dtype == "object":
                feature.string_stats.CopyFrom(
                    statistics_pb2.StringStatistics(
                        common_stats=common_stats,
                        avg_length=dask_stats[col]["avg_str_length"],
                        unique=dask_stats[col]["nunique"].item(),
                    )
                )
                feature.type = 2

                ranks = feature.string_stats.rank_histogram
                for ind, (val, freq) in enumerate(
                    dask_stats[col]["top_values"].to_pandas().items()
                ):
                    f = feature.string_stats.top_values.add()
                    f.value = val
                    f.frequency = freq
                    b = ranks.buckets.add()
                    b.CopyFrom(
                        statistics_pb2.RankHistogram.Bucket(
                            low_rank=ind, high_rank=ind, label=val, sample_count=freq
                        )
                    )

        if self.cross_columns:
            corr, cov = dask_stats["corr"], dask_stats["cov"]
            for (path_x, cov_val), (_, corr_val) in zip(cov.items(), corr.items()):
                for (path_y, covariance), correlation in zip(cov_val.items(), corr_val.items()):
                    cross = statistics_pb2.CrossFeatureStatistics(
                        path_x=path_pb2.Path(step=[path_x]),
                        path_y=path_pb2.Path(step=[path_y]),
                        num_cross_stats=statistics_pb2.NumericCrossStatistics(
                            correlation=float(correlation), covariance=float(covariance)
                        ),
                    )
                    c = self.stats.cross_features.add()
                    c.CopyFrom(cross)

        if self.stats_path:
            self.stats_list().save(self.output_path)

    def clear(self):
        self.schema = None

    def stats_list(self, others=None) -> vis_statistics.DatasetCollectionStatistics:
        if not others:
            others = []
        data = statistics_pb2.DatasetFeatureStatisticsList()

        d = data.datasets.add()
        d.CopyFrom(self.stats)

        for d in others:
            x = data.datasets.add()
            x.CopyFrom(d)

        return vis_statistics.DatasetCollectionStatistics(data)

    def to_schema(self, tags_by_column):
        self.tags_by_column = tags_by_column

        return prepare_schema(
            self.stats, self.col_names, self.col_dtypes, tags_by_column=tags_by_column
        )

    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
