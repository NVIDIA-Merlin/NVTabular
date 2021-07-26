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

import dask.dataframe as dd
from nvtx import annotate

from nvtabular.dispatch import DataFrameType

from ..column import Columns
from .base import Operator
from .stat_operator import StatOperator

LOG = logging.getLogger("nvtabular")


class DataSchema(StatOperator):
    SCHEMA_FILE_NAME = "schema.pb"

    def __init__(self, tags_by_column, output_path=None):
        super().__init__()
        self.schema_path = os.path.join(output_path, self.SCHEMA_FILE_NAME) if output_path else None
        self.col_names = []
        self.col_types = []
        self.col_dtypes = []
        self.tags_by_column = tags_by_column
        self.schema = None

    @classmethod
    def calculate_on_dataset(cls, dataset, tags_by_column, output_path=None, client=None):
        from nvtabular.column_group import ColumnGroup
        from nvtabular.workflow import Workflow

        stats = cls(tags_by_column, output_path=output_path)

        col_group = ColumnGroup([])

        for column in dataset.columns:
            col_group += ColumnGroup(column, tags=tags_by_column.get(column))

        workflow = Workflow(col_group >> stats, client=client)
        workflow.fit(dataset)

        return stats.schema

    @classmethod
    def load(cls, directory):
        from tensorflow_metadata.proto.v0 import schema_pb2

        schema_file = os.path.join(directory, cls.SCHEMA_FILE_NAME)

        LOG.info("Loading schema from %s", schema_file)

        if not os.path.exists(schema_file):
            return None

        schema = schema_pb2.Schema()
        with open(schema_file, "rb") as f:
            schema.ParseFromString(f.read())

        return schema

    def transform(self, columns: Columns, df: DataFrameType) -> DataFrameType:
        return df

    @annotate("Schema_fit", color="green", domain="nvt_python")
    def fit(self, columns: Columns, ddf: dd.DataFrame):
        dask_stats = {}

        ddf_dtypes = ddf.head(1)

        # For each column, calculate the stats
        for col in columns:
            dask_stats[col] = {}
            self.col_names.append(col)
            # Get dtype for all
            dtype = ddf_dtypes[col].dtype
            self.col_dtypes.append(ddf_dtypes[col].dtype)

            domain = ddf[col]

            if str(dtype) == "list":
                domain = ddf[col].map_partitions(lambda x: x.list.leaves, meta=("x", int))
                lengths = ddf[col].map_partitions(lambda x: x.list.len(), meta=("x", int))
                dask_stats[col]["min_length"] = lengths.min()
                dask_stats[col]["max_length"] = lengths.max()
                dtype = dtype.leaf_type

            if str(dtype) in ["int8", "int32", "int64", "float32", "float64"]:
                dask_stats[col]["min"] = domain.min()
                dask_stats[col]["max"] = domain.max()

        return dask_stats

    def fit_finalize(self, stats):
        return self.prepare_schema(stats)

    def prepare_schema(self, stats):
        from tensorflow_metadata.proto.v0 import schema_pb2

        dask_stats = stats

        self.schema = schema_pb2.Schema()

        for i, col in enumerate(self.col_names):
            dtype = self.col_dtypes[i]
            tags = self.tags_by_column.get(col, [])

            feature = self.schema.feature.add()
            feature.name = col
            feature.annotation.CopyFrom(schema_pb2.Annotation(tag=tags))

            if str(dtype) == "list":
                dtype = dtype.leaf_type
                min_length = dask_stats[col]["min_length"].item()
                max_length = dask_stats[col]["max_length"].item()
                if min_length == max_length:
                    shape = schema_pb2.FixedShape()
                    dim = shape.dim.add()
                    dim.size = min_length
                    feature.shape.CopyFrom(shape)
                else:
                    feature.value_count.CopyFrom(
                        schema_pb2.ValueCount(min=min_length, max=max_length)
                    )

            if str(dtype) in ["float32", "float64"]:
                feature.float_domain.CopyFrom(
                    schema_pb2.FloatDomain(
                        name=col,
                        min=dask_stats[col]["min"].item(),
                        max=dask_stats[col]["max"].item(),
                    )
                )
                feature.type = 3
            elif str(dtype) in ["int8", "int32", "int64"]:
                feature.int_domain.CopyFrom(
                    schema_pb2.IntDomain(
                        name=col,
                        min=dask_stats[col]["min"].item(),
                        max=dask_stats[col]["max"].item(),
                        is_categorical="categorical" in tags,
                    )
                )
                feature.type = 2

        if self.schema_path:
            self.save(self.schema)

    def clear(self):
        self.schema = None

    def save(self, schema):
        with open(self.schema_path, "wb") as f:
            f.write(schema.SerializeToString())

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
