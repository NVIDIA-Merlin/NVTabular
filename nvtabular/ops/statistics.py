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
import base64
import os

import dask.array as da
import dask.dataframe as dd
import numpy as np
from nvtx import annotate

from nvtabular.dispatch import DataFrameType
from .schema import Schema

from .base import ColumnNames, Operator
from .stat_operator import StatOperator


class Statistics(Schema):
    STATS_FILE_NAME = "stats.pb"

    def __init__(self, output_path=None, name="data"):
        super(StatOperator, self).__init__()
        self.name = name
        self.stats_path = os.path.join(output_path, self.STATS_FILE_NAME) if output_path else None
        self.output_path = output_path
        self.col_names = []
        self.col_types = []
        self.col_dtypes = []
        self.stats = None

    @classmethod
    def calculate_on_dataset(cls, dataset, column_group=None, output_path=None, client=None):
        from nvtabular.workflow import Workflow

        column_group = column_group or list(dataset.to_ddf().columns)
        stats = cls(output_path=output_path)

        workflow = Workflow(column_group >> stats, output_path, client=client)
        workflow.fit(dataset, save_workflow=False)

        return stats

    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        return df

    @annotate("Statistics_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dd.DataFrame):
        dask_stats = {}

        ddf_dtypes = ddf.head(1)
        num_examples = len(ddf)

        #         dask_stats["median"] = ddf[columns].quantile(q=0.5, method="dask")

        # For each column, calculate the stats
        for col in columns:
            dask_stats[col] = {}
            self.col_names.append(col)
            # Get dtype for all
            dtype = ddf_dtypes[col].dtype
            self.col_dtypes.append(dtype)

            dask_stats[col]["num_missing"] = len(ddf[col][ddf[col].isnull()])
            dask_stats[col]["num_non_missing"] = ddf[col].count()

            # Get cardinality for cats
            if dtype == np.object:
                dask_stats[col]["nunique"] = ddf[col].nunique()
                str_len = ddf[col].map_partitions(lambda x: x.str.len(), meta=("x", int))
                dask_stats[col]["avg_length"] = str_len.mean()
                dask_stats[col]["top_values"] = ddf[col].value_counts().nlargest(n=50)

            elif dtype in [np.int32, np.int64, np.float32]:
                # Get various stats
                dask_stats[col]["min"] = ddf[col].min()
                dask_stats[col]["max"] = ddf[col].max()
                dask_stats[col]["mean"] = ddf[col].mean()
                dask_stats[col]["std"] = ddf[col].std()

                dask_stats[col]["num_zeroes"] = (ddf[col] == 0).sum()

                h, bins = da.histogram(ddf[col].to_dask_array(), 10, range=[ddf[col].min(), ddf[col].max()])
                dask_stats[col]["histogram"] = h
                dask_stats[col]["histogram_bins"] = bins

        return num_examples, dask_stats

    def fit_finalize(self, stats):
        from tensorflow_metadata.proto.v0 import statistics_pb2
        num_examples, dask_stats = stats

        self.stats = statistics_pb2.DatasetFeatureStatistics(name=self.name, num_examples=num_examples)

        for i, col in enumerate(self.col_names):
            dtype = self.col_dtypes[i]

            feature = self.stats.features.add()
            feature.name = col

            common_stats = statistics_pb2.CommonStatistics(
                num_non_missing=int(dask_stats[col]["num_non_missing"]),
                num_missing=int(dask_stats[col]["num_missing"]),
                min_num_values=1,
                max_num_values=1,
                avg_num_values=1.0,
            )

            if dtype in [np.int32, np.int64, np.float32]:
                hist = statistics_pb2.Histogram(type=0)
                h, bins = dask_stats[col]["histogram"], dask_stats[col]["histogram_bins"]
                for i in range(len(h)):
                    bucket = hist.buckets.add()
                    bucket.low_value = bins[i]
                    bucket.high_value = bins[i + 1]
                    bucket.sample_count = float(h[i])

                feature.num_stats.CopyFrom(statistics_pb2.NumericStatistics(
                    min=dask_stats[col]["min"].item(),
                    max=dask_stats[col]["max"].item(),
                    histograms=[hist],
                    common_stats=common_stats,
                    #                     median=1.0,
                    #                     median=float(dask_stats[col]["median"]),
                    mean=dask_stats[col]["mean"].item(),
                    std_dev=dask_stats[col]["std"].item(),
                    num_zeros=dask_stats[col]["num_zeroes"].item(),
                ))
                feature.type = 1 if np.issubdtype(dtype, np.floating) else 0
            elif dtype == np.object:
                feature.string_stats.CopyFrom(statistics_pb2.StringStatistics(
                    common_stats=common_stats,
                    avg_length=dask_stats[col]["avg_length"],
                    unique=dask_stats[col]["nunique"].item()
                ))

                ranks = feature.string_stats.rank_histogram
                for ind, (val, freq) in enumerate(dask_stats[col]["top_values"].to_pandas().items()):
                    f = feature.string_stats.top_values.add()
                    f.value = val
                    f.frequency = freq
                    b = ranks.buckets.add()
                    b.CopyFrom(
                        statistics_pb2.RankHistogram.Bucket(low_rank=ind, high_rank=ind, label=val, sample_count=freq))

        if self.stats_path:
            self.stats_list().save(self.output_path)

    def clear(self):
        self.schema = None

    def stats_list(self, others=None) -> "DatasetCollectionStatistics":
        if not others:
            others = []
        from tensorflow_metadata.proto.v0 import statistics_pb2

        data = statistics_pb2.DatasetFeatureStatisticsList()

        d = data.datasets.add()
        d.CopyFrom(self.stats)

        for d in others:
            x = data.datasets.add()
            x.CopyFrom(d)

        return DatasetCollectionStatistics(data)

    def to_schema(self, tags_by_column):
        self.tags_by_column = tags_by_column

        return self.prepare_schema(self.stats)

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__


class DatasetCollectionStatistics(object):
    HTML_TEMPLATE = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
<link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
<facets-overview id="elem"></facets-overview>
<script>
 document.querySelector("#elem").protoInput = "{protostr}";
</script>"""

    def __init__(self, dataset_feature_statistics_list) -> None:
        super().__init__()
        self.stats = dataset_feature_statistics_list

    def display(self):
        from IPython.core.display import display, HTML

        return display(HTML(self.to_html()))

    def to_html(self):
        protostr = self.to_proto_string(self.stats)
        html = self.HTML_TEMPLATE.format(protostr=protostr)

        return html

    def save_to_html(self, output_dir, file_name="stats.html"):
        with open(os.path.join(output_dir, file_name), "w") as html_file:
            html_file.write(self.to_html())

    def to_proto_string(self, inputs):
        return base64.b64encode(inputs.SerializeToString()).decode("utf-8")

    def save(self, output_dir, file_name=Statistics.STATS_FILE_NAME):
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "wb") as f:
            f.write(self.stats.SerializeToString())

        self.save_to_html(output_dir)
