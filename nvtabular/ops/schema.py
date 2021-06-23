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
import dask.dataframe as dd
import numpy as np
import os
from nvtx import annotate

from nvtabular.dispatch import DataFrameType

from .operator import ColumnNames, Operator
from .stat_operator import StatOperator
from .. import Dataset
from ..column_group import ColumnGroup
from ..workflow import Workflow


class Schema(StatOperator):
    def __init__(self, column_group: ColumnGroup, output_path=None):
        super().__init__()
        self.schema_path = os.path.join(output_path, "schema.pb") if output_path else None
        self.col_names = []
        self.col_types = []
        self.col_dtypes = []
        self.schema = None
        self.column_group = column_group
        self.tags_by_column = column_group.tags_by_column()

    @classmethod
    def calculate_on_dataset(cls, dataset: Dataset, column_group: ColumnGroup, output_path=None, client=None):
        stats = cls(column_group, output_path=output_path)

        new_col_group = ColumnGroup([])
        tags_by_column = column_group.tags_by_column()

        for key, val in tags_by_column.items():
            new_col_group += ColumnGroup(key, tags=val)

        new_col_group >> stats
        workflow = Workflow(new_col_group, output_path, client=client)
        workflow.fit(dataset)

        return stats.schema

    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        return df

    @annotate("DataStats_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dd.DataFrame):
        dask_stats = {}

        ddf_dtypes = ddf.head(1)

        # For each column, calculate the stats
        for col in columns:
            dask_stats[col] = {}
            self.col_names.append(col)
            # Get dtype for all
            dtype = ddf_dtypes[col].dtype
            self.col_dtypes.append(dtype)

            # Identify column type
            if np.issubdtype(dtype, np.floating):
                col_type = "conts"
            else:
                col_type = "cats"
            self.col_types.append(dtype)

            # Get cardinality for cats
            if col_type == "cats":
                dask_stats[col]["cardinality"] = ddf[col].nunique()

            # if string, replace string for their lengths for the rest of the computations
            if dtype == "object":
                ddf[col] = ddf[col].map_partitions(lambda x: x.str.len(), meta=("x", int))
            # Add list support when cudf supports it:
            # https://github.com/rapidsai/cudf/issues/7157
            # elif col_type == "cat_mh":
            #    ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())

            # Get min,max, and mean
            dask_stats[col]["min"] = ddf[col].min()
            dask_stats[col]["max"] = ddf[col].max()

        return dask_stats

    def fit_finalize(self, stats):
        from tensorflow_metadata.proto.v0 import schema_pb2
        dask_stats = stats

        schema = schema_pb2.Schema()

        for i, col in enumerate(self.col_names):
            feature = schema.feature.add()
            dtype = str(self.col_dtypes[i])
            tags = self.tags_by_column.get(col, [])

            if dtype == np.float32:
                feature.float_domain.CopyFrom(schema_pb2.FloatDomain(
                    name=col,
                    min=dask_stats[col]["min"].item(),
                    max=dask_stats[col]["max"].item()
                ))
                feature.type = 3
            elif dtype in [np.int32, np.int64]:
                feature.int_domain.CopyFrom(schema_pb2.IntDomain(
                    name=col,
                    min=dask_stats[col]["min"].item(),
                    max=dask_stats[col]["max"].item(),
                    is_categorical="categorical" in tags
                ))
                feature.type = 2

        if self.schema_path:
            with open(self.schema_path, "wb") as f:
                f.write(schema.SerializeToString())
        self.schema = schema

    def clear(self):
        self.output = {}

    # def save(self):
    #     from tensorflow_metadata.proto.v0 import schema_pb2
    #
    #     column_group = self.workflow.column_group
    #     tags_by_column = column_group.tags_by_column()
    #
    #     schema = schema_pb2.Schema()
    #     ddf_dtypes = dict(ddf.dtypes)
    #
    #     for f in list(ddf.columns):
    #         if f.startswith("Unnamed"):
    #             continue
    #
    #         feature_dtype = ddf_dtypes[f]
    #
    #         feature = schema.feature.add()
    #         feature.name = f
    #         tags = tags_by_column.get(f, [])
    #         feature.annotation.CopyFrom(schema_pb2.Annotation(tag=tags))
    #
    #         if feature_dtype == np.float32:
    #             feature.float_domain.CopyFrom(schema_pb2.FloatDomain(
    #                 name=f,
    #                 min=ddf[f].min().compute(),
    #                 max=ddf[f].max().compute()
    #             ))
    #             feature.type = 3
    #         elif feature_dtype in [np.int32, np.int64]:
    #             feature.int_domain.CopyFrom(schema_pb2.IntDomain(
    #                 name=f,
    #                 min=ddf[f].min().compute(),
    #                 max=ddf[f].max().compute(),
    #                 is_categorical="categorical" in tags
    #             ))
    #             feature.type = 2
    #
    #     with open(schema_file, "wb") as f:
    #         f.write(schema.SerializeToString())

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
