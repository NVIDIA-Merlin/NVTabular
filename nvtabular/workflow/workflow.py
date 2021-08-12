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
import json
import logging
import os
import sys
import time
import warnings
from typing import TYPE_CHECKING, Optional

import cloudpickle

from nvtabular.columns.schema import ColumnSchema, DatasetSchema

try:
    import cudf
except ImportError:
    cudf = None
import dask
import pandas as pd
from dask.core import flatten

from nvtabular.dispatch import _concat_columns
from nvtabular.io.dataset import Dataset
from nvtabular.ops import StatOperator
from nvtabular.ops.operator import ColumnSelector
from nvtabular.utils import _ensure_optimize_dataframe_graph, global_dask_client
from nvtabular.worker import clean_worker_cache
from nvtabular.workflow.node import WorkflowNode, _merge_add_nodes, iter_nodes

LOG = logging.getLogger("nvtabular")


if TYPE_CHECKING:
    import distributed


class Workflow:
    """
    The Workflow class applies a graph of operations onto a dataset, letting you transform
    datasets to do feature engineering and preprocessing operations. This class follows an API
    similar to Transformers in sklearn: we first ``fit`` the workflow by calculating statistics
    on the dataset, and then once fit we can ``transform`` datasets by applying these statistics.

    Example usage::

        # define a graph of operations
        cat_features = CAT_COLUMNS >> nvtabular.ops.Categorify()
        cont_features = CONT_COLUMNS >> nvtabular.ops.FillMissing() >> nvtabular.ops.Normalize()
        workflow = nvtabular.Workflow(cat_features + cont_features + "label")

        # calculate statistics on the training dataset
        workflow.fit(nvtabular.io.Dataset(TRAIN_PATH))

        # transform the training and validation datasets and write out as parquet
        workflow.transform(nvtabular.io.Dataset(TRAIN_PATH)).to_parquet(output_path=TRAIN_OUT_PATH)
        workflow.transform(nvtabular.io.Dataset(VALID_PATH)).to_parquet(output_path=VALID_OUT_PATH)

    Parameters
    ----------
    output_node: WorkflowNode
        The last node in the graph of operators this workflow should apply
    client: distributed.Client, optional
        The Dask distributed client to use for multi-gpu processing and multi-node processing
    """

    def __init__(self, output_node: WorkflowNode, client: Optional["distributed.Client"] = None):
        self.output_node = _merge_add_nodes(output_node)
        self.client = client
        self.input_schema = None
        self.input_dtypes = None
        self.output_dtypes = None

        # Warn user if there is an unused global
        # Dask client available
        if global_dask_client(self.client):
            warnings.warn(
                "A global dask.distributed client has been detected, but the "
                "single-threaded scheduler will be used for execution. Please "
                "use the `client` argument to initialize a `Workflow` object "
                "with distributed-execution enabled."
            )

    def transform(self, dataset: Dataset) -> Dataset:
        """Transforms the dataset by applying the graph of operators to it. Requires the ``fit``
        method to have already been called, or calculated statistics to be loaded from disk

        This method returns a Dataset object, with the transformations lazily loaded. None
        of the actual computation will happen until the produced Dataset is consumed, or
        written out to disk.

        Parameters
        -----------
        dataset: Dataset

        Returns
        -------
        Dataset
        """
        self._clear_worker_cache()
        ddf = dataset.to_ddf(columns=self._input_columns)
        return Dataset(
            _transform_ddf(ddf, self.output_node, self.output_dtypes),
            client=self.client,
            cpu=dataset.cpu,
            base_dataset=dataset.base_dataset,
        )

    @property
    def output_schema(self):
        return self.output_node.output_schema

    def fit_schema(self, input_schema: DatasetSchema):
        schemaless_nodes = {
            node: _get_schemaless_nodes(node.parents)
            for node in _get_schemaless_nodes([self.output_node])
        }

        while schemaless_nodes:
            # get all the Operators with no outstanding dependencies
            current_phase = [
                node for node, dependencies in schemaless_nodes.items() if not dependencies
            ]
            if not current_phase:
                # this shouldn't happen, but lets not infinite loop just in case
                raise RuntimeError("failed to find dependency-free Operator to compute schema for")

            processed_nodes = []
            for node in current_phase:
                if not node.parents:
                    node.compute_schemas(input_schema)
                else:
                    combined_schema = sum(
                        [parent.output_schema for parent in node.parents if parent.output_schema],
                        DatasetSchema(),
                    )

                    node.compute_schemas(combined_schema)

                processed_nodes.append(node)

            # Remove all the operators we processed in this phase, and remove
            # from the dependencies of other ops too
            for schemaless_node in current_phase:
                schemaless_nodes.pop(schemaless_node)
            for dependencies in schemaless_nodes.values():
                dependencies.difference_update(current_phase)

    def fit(self, dataset: Dataset, input_schema: DatasetSchema = None):
        """Calculates statistics for this workflow on the input dataset

        Parameters
        -----------
        dataset: Dataset
            The input dataset to calculate statistics for. If there is a train/test split this
            data should be the training dataset only.
        """
        self._clear_worker_cache()

        # Compute which input columns we need to load from the source dataset
        if input_schema:
            # TODO: Validate that the dataset matches the schema (if present)
            self.input_schema = input_schema
            input_columns = self._input_columns
        else:
            input_columns = None

        ddf = dataset.to_ddf(columns=input_columns)

        # Create a schema for the dataset if none was supplied
        # TODO: Extract this to the dataset __init__ method?
        if not input_schema:
            col_schemas = []
            for column_name in ddf.columns:
                col_schemas.append(ColumnSchema(column_name))
            self.input_schema = DatasetSchema(col_schemas)

        self.fit_schema(self.input_schema)

        # Get a dictionary mapping all StatOperators we need to fit to a set of any dependant
        # StatOperators (having StatOperators that depend on the output of other StatOperators
        # means that will have multiple phases in the fit cycle here)
        stat_ops = {op: _get_stat_ops(op.parents) for op in _get_stat_ops([self.output_node])}

        while stat_ops:
            # get all the StatOperators that we can currently call fit on (no outstanding
            # dependencies)
            current_phase = [op for op, dependencies in stat_ops.items() if not dependencies]
            if not current_phase:
                # this shouldn't happen, but lets not infinite loop just in case
                raise RuntimeError("failed to find dependency-free StatOperator to fit")

            stats, ops = [], []
            for workflow_node in current_phase:
                # apply transforms necessary for the inputs to the current column group, ignoring
                # the transforms from the statop itself
                transformed_ddf = _ensure_optimize_dataframe_graph(
                    ddf=_transform_ddf(ddf, workflow_node.parents)
                )

                op = workflow_node.op
                try:
                    col_selector = ColumnSelector(workflow_node.input_column_names)
                    stats.append(op.fit(col_selector, transformed_ddf))
                    ops.append(op)
                except Exception:
                    LOG.exception("Failed to fit operator %s", workflow_node.op)
                    raise

            if self.client:
                results = [r.result() for r in self.client.compute(stats)]
            else:
                results = dask.compute(stats, scheduler="synchronous")[0]

            for computed_stats, op in zip(results, ops):
                op.fit_finalize(computed_stats)

            # Remove all the operators we processed in this phase, and remove
            # from the dependencies of other ops too
            for stat_op in current_phase:
                stat_ops.pop(stat_op)
            for dependencies in stat_ops.values():
                dependencies.difference_update(current_phase)

        # hack: store input/output dtypes here. We should have complete dtype
        # information for each operator (like we do for column names), but as
        # an interim solution this gets us what we need.
        input_dtypes = dataset.to_ddf()[self._input_columns].dtypes
        self.input_dtypes = dict(zip(input_dtypes.index, input_dtypes))
        output_dtypes = self.transform(dataset).sample_dtypes()
        self.output_dtypes = dict(zip(output_dtypes.index, output_dtypes))

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Convenience method to both fit the workflow and transform the dataset in a single
        call. Equivalent to calling ``workflow.fit(dataset)`` followed by
        ``workflow.transform(dataset)``

        Parameters
        -----------
        dataset: Dataset

        Returns
        -------
        Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

    def save(self, path):
        """Save this workflow to disk

        Parameters
        ----------
        path: str
            The path to save the workflow to
        """
        # avoid a circular import getting the version
        from nvtabular import __version__ as nvt_version

        os.makedirs(path, exist_ok=True)

        # point all stat ops to store intermediate output (parquet etc) at the path
        # this lets us easily bundle
        for stat in _get_stat_ops([self.output_node]):
            stat.op.set_storage_path(path, copy=True)

        # generate a file of all versions used to generate this bundle
        lib = cudf if cudf else pd
        with open(os.path.join(path, "metadata.json"), "w") as o:
            json.dump(
                {
                    "versions": {
                        "nvtabular": nvt_version,
                        lib.__name__: lib.__version__,
                        "python": sys.version,
                    },
                    "generated_timestamp": int(time.time()),
                },
                o,
            )

        # dump out the full workflow (graph/stats/operators etc) using cloudpickle
        with open(os.path.join(path, "workflow.pkl"), "wb") as o:
            cloudpickle.dump(self, o)

    @classmethod
    def load(cls, path, client=None):
        """Load up a saved workflow object from disk

        Parameters
        ----------
        path: str
            The path to load the workflow from
        client: distributed.Client, optional
            The Dask distributed client to use for multi-gpu processing and multi-node processing

        Returns
        -------
            Workflow
        """
        # avoid a circular import getting the version
        from nvtabular import __version__ as nvt_version

        # check version information from the metadata blob, and warn if we have a mismatch
        meta = json.load(open(os.path.join(path, "metadata.json")))

        def parse_version(version):
            return version.split(".")[:2]

        def check_version(stored, current, name):
            if parse_version(stored) != parse_version(current):
                warnings.warn(
                    f"Loading workflow generated with {name} version {stored} "
                    f"- but we are running {name} {current}. This might cause issues"
                )

        # make sure we don't have any major/minor version conflicts between the stored worklflow
        # and the current environment
        lib = cudf if cudf else pd
        versions = meta["versions"]
        check_version(versions["nvtabular"], nvt_version, "nvtabular")
        check_version(versions["python"], sys.version, "python")

        if lib.__name__ in versions:
            check_version(versions[lib.__name__], lib.__version__, lib.__name__)
        else:
            expected = "GPU" if "cudf" in versions else "CPU"
            warnings.warn(f"Loading workflow generated on {expected}")

        # load up the workflow object di
        workflow = cloudpickle.load(open(os.path.join(path, "workflow.pkl"), "rb"))
        workflow.client = client

        # we might have been copied since saving, update all the stat ops
        # with the new path to their storage locations
        for stat in _get_stat_ops([workflow.output_node]):
            stat.op.set_storage_path(path, copy=False)

        return workflow

    def __getstate__(self):
        # dask client objects aren't picklable - exclude from saved representation
        return {k: v for k, v in self.__dict__.items() if k != "client"}

    def clear_stats(self):
        for stat in _get_stat_ops([self.output_node]):
            stat.op.clear()

    @property
    def _input_columns(self):
        if not self.input_schema:
            raise ValueError("Input columns can't be computed until the input schema is provided.")

        selectors = self._input_selectors()
        input_columns = set()
        for selector in selectors:
            for column_name in selector.names:
                col_schema = self.input_schema[column_name]
                input_columns.add(col_schema)

        return list(input_columns)

    @property
    def _input_selectors(self):
        return [node.selector for node in self._input_nodes]

    @property
    def _input_nodes(self):
        return set(node for node in iter_nodes([self.output_node]) if not node.parents)

    def _clear_worker_cache(self):
        # Clear worker caches to be "safe"
        if self.client:
            self.client.run(clean_worker_cache)
        else:
            clean_worker_cache()


def _transform_ddf(ddf, workflow_nodes, meta=None):
    if isinstance(workflow_nodes, WorkflowNode):
        workflow_nodes = [workflow_nodes]

    columns = list(flatten(cg.flattened_columns for cg in workflow_nodes))

    # Check if we are only selecting columns (no transforms).
    # If so, we should perform column selection at the ddf level.
    # Otherwise, Dask will not push the column selection into the
    # IO function.
    if all((c.op is None and not c.parents) for c in workflow_nodes):
        return ddf[_get_unique(columns)]

    if isinstance(meta, dict) and isinstance(ddf._meta, pd.DataFrame):
        dtypes = meta
        meta = type(ddf._meta)({k: [] for k in columns})
        for column, dtype in dtypes.items():
            meta[column] = meta[column].astype(dtype)

    elif not meta:
        # TODO: constructing meta like this loses dtype information on the ddf
        # and sets it all to 'float64'. We should propogate dtype information along
        # with column names in the columngroup graph. This currently only
        # happesn during intermediate 'fit' transforms, so as long as statoperators
        # don't require dtype information on the DDF this doesn't matter all that much
        meta = type(ddf._meta)({k: [] for k in columns})

    return ddf.map_partitions(
        _transform_partition,
        workflow_nodes,
        meta=meta,
    )


def _get_stat_ops(nodes):
    return set(node for node in iter_nodes(nodes) if isinstance(node.op, StatOperator))


def _get_schemaless_nodes(nodes):
    return set(node for node in iter_nodes(nodes) if node.input_schema is None)


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())


def _transform_partition(root_df, workflow_nodes):
    """Transforms a single partition by appyling all operators in a WorkflowNode"""
    output = None
    for workflow_node in workflow_nodes:
        unique_flattened_cols = _get_unique(workflow_node.flattened_columns)
        # collect dependencies recursively if we have parents
        if workflow_node.parents:
            df = None
            columns = None
            for parent in workflow_node.parents:
                unique_flattened_cols_parent = _get_unique(parent.flattened_columns)
                parent_df = _transform_partition(root_df, [parent])
                if df is None or not len(df):
                    df = parent_df[unique_flattened_cols_parent]
                    columns = set(unique_flattened_cols_parent)
                else:
                    new_columns = set(unique_flattened_cols_parent) - columns
                    df = _concat_columns([df, parent_df[list(new_columns)]])
                    columns.update(new_columns)
        else:
            # otherwise select the input from the root df
            df = root_df[unique_flattened_cols]

        # apply the operator if necessary
        if workflow_node.op:
            try:
                col_selector = ColumnSelector(workflow_node.input_column_names)
                df = workflow_node.op.transform(col_selector, df)
            except Exception:
                LOG.exception("Failed to transform operator %s", workflow_node.op)
                raise
            if df is None:
                raise RuntimeError(
                    "Operator %s didn't return a value during transform" % workflow_node.op
                )

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        if output is None:
            output = df[unique_flattened_cols]
        else:
            output = _concat_columns([output, df[unique_flattened_cols]])

    return output
