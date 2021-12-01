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
import sys
import time
import warnings
from typing import TYPE_CHECKING, Optional

import cloudpickle
import fsspec

try:
    import cudf
except ImportError:
    cudf = None
import dask
import pandas as pd
from dask.core import flatten

import nvtabular
from nvtabular.dispatch import _concat_columns
from nvtabular.graph.node import iter_nodes
from nvtabular.graph.schema import Schema
from nvtabular.io.dataset import Dataset
from nvtabular.ops import StatOperator
from nvtabular.utils import _ensure_optimize_dataframe_graph, global_dask_client
from nvtabular.worker import clean_worker_cache
from nvtabular.workflow.node import WorkflowNode

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
        self.output_node = output_node
        self.client = client
        self.input_dtypes = None
        self.output_dtypes = None
        self.output_schema = None

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

        if not self.output_schema:
            self.fit_schema(dataset.schema)

        ddf = dataset.to_ddf(columns=self._input_columns())
        return Dataset(
            _transform_ddf(ddf, self.output_node, self.output_dtypes),
            client=self.client,
            cpu=dataset.cpu,
            base_dataset=dataset.base_dataset,
            schema=self.output_schema,
        )

    def fit_schema(self, input_schema: Schema) -> "Workflow":
        schemaless_nodes = {
            node: _get_schemaless_nodes(node.parents_with_dependencies)
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
                        Schema(),
                    )
                    # we want to update the input_schema with new values
                    # from combined schema
                    combined_schema = input_schema + combined_schema
                    node.compute_schemas(combined_schema)

                processed_nodes.append(node)

            # Remove all the operators we processed in this phase, and remove
            # from the dependencies of other ops too
            for schemaless_node in current_phase:
                schemaless_nodes.pop(schemaless_node)
            for dependencies in schemaless_nodes.values():
                dependencies.difference_update(current_phase)

        self.output_schema = self.output_node.output_schema

        return self

    def fit(self, dataset: Dataset) -> "Workflow":
        """Calculates statistics for this workflow on the input dataset

        Parameters
        -----------
        dataset: Dataset
            The input dataset to calculate statistics for. If there is a train/test split this
            data should be the training dataset only.
        """
        self._clear_worker_cache()

        if not self.output_schema:
            self.fit_schema(dataset.schema)

        ddf = dataset.to_ddf(columns=self._input_columns())

        # Get a dictionary mapping all StatOperators we need to fit to a set of any dependent
        # StatOperators (having StatOperators that depend on the output of other StatOperators
        # means that will have multiple phases in the fit cycle here)
        stat_ops = {
            op: _get_stat_ops(op.parents_with_dependencies)
            for op in _get_stat_ops([self.output_node])
        }

        while stat_ops:
            # get all the StatOperators that we can currently call fit on (no outstanding
            # dependencies)
            current_phase = [op for op, dependencies in stat_ops.items() if not dependencies]
            if not current_phase:
                # this shouldn't happen, but lets not infinite loop just in case
                raise RuntimeError("failed to find dependency-free StatOperator to fit")

            stats, ops = [], []
            for workflow_node in current_phase:
                # Check for additional input columns that aren't generated by parents
                addl_input_cols = set()
                if workflow_node.parents:
                    upstream_output_cols = sum(
                        [
                            upstream.output_columns
                            for upstream in workflow_node.parents_with_dependencies
                        ],
                        nvtabular.ColumnSelector(),
                    )
                    addl_input_cols = set(workflow_node.input_columns.names) - set(
                        upstream_output_cols.names
                    )

                # apply transforms necessary for the inputs to the current column group, ignoring
                # the transforms from the statop itself
                transformed_ddf = _ensure_optimize_dataframe_graph(
                    ddf=_transform_ddf(
                        ddf,
                        workflow_node.parents_with_dependencies,
                        additional_columns=addl_input_cols,
                    )
                )

                op = workflow_node.op
                try:
                    stats.append(op.fit(workflow_node.input_columns, transformed_ddf))
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
        input_dtypes = dataset.to_ddf()[self._input_columns()].dtypes
        self.input_dtypes = dict(zip(input_dtypes.index, input_dtypes))
        output_dtypes = self.transform(dataset).sample_dtypes()
        self.output_dtypes = dict(zip(output_dtypes.index, output_dtypes))

        self._zero_output_schemas()
        self.fit_schema(dataset.schema)
        return self

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

        fs = fsspec.get_fs_token_paths(path)[0]

        fs.makedirs(path, exist_ok=True)

        # point all stat ops to store intermediate output (parquet etc) at the path
        # this lets us easily bundle
        for stat in _get_stat_ops([self.output_node]):
            stat.op.set_storage_path(path, copy=True)

        # generate a file of all versions used to generate this bundle
        lib = cudf if cudf else pd
        with fs.open(fs.sep.join([path, "metadata.json"]), "w") as o:
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
        with fs.open(fs.sep.join([path, "workflow.pkl"]), "wb") as o:
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

        fs = fsspec.get_fs_token_paths(path)[0]

        # check version information from the metadata blob, and warn if we have a mismatch
        meta = json.load(fs.open(fs.sep.join([path, "metadata.json"])))

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
        workflow = cloudpickle.load(fs.open(fs.sep.join([path, "workflow.pkl"]), "rb"))
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

    def _input_columns(self):
        input_cols = []
        for node in iter_nodes([self.output_node]):
            upstream_output_cols = []

            for upstream_node in node.parents_with_dependencies:
                upstream_output_cols += upstream_node.output_columns.names

            upstream_output_cols = _get_unique(upstream_output_cols)
            input_cols += list(set(node.input_columns.names) - set(upstream_output_cols))

        return _get_unique(input_cols)

    def _clear_worker_cache(self):
        # Clear worker caches to be "safe"
        if self.client:
            self.client.run(clean_worker_cache)
        else:
            clean_worker_cache()

    def _zero_output_schemas(self):
        """
        Zero out all schemas in order to rerun fit schema after operators
        have run fit and have stats to add to schema.
        """
        for node in iter_nodes([self.output_node]):
            node.output_schema = None
            node.input_schema = None


def _transform_ddf(ddf, workflow_nodes, meta=None, additional_columns=None):
    # Check if we are only selecting columns (no transforms).
    # If so, we should perform column selection at the ddf level.
    # Otherwise, Dask will not push the column selection into the
    # IO function.
    if not workflow_nodes:
        return ddf[_get_unique(additional_columns)] if additional_columns else ddf

    if isinstance(workflow_nodes, WorkflowNode):
        workflow_nodes = [workflow_nodes]

    columns = list(flatten(wfn.output_columns.names for wfn in workflow_nodes))
    columns += additional_columns if additional_columns else []

    if isinstance(meta, dict) and isinstance(ddf._meta, pd.DataFrame):
        dtypes = meta
        meta = type(ddf._meta)({k: [] for k in columns})
        for column, dtype in dtypes.items():
            meta[column] = meta[column].astype(dtype)

    elif not meta:
        # TODO: constructing meta like this loses dtype information on the ddf
        # and sets it all to 'float64'. We should propagate dtype information along
        # with column names in the columngroup graph. This currently only
        # happesn during intermediate 'fit' transforms, so as long as statoperators
        # don't require dtype information on the DDF this doesn't matter all that much
        meta = type(ddf._meta)({k: [] for k in columns})

    return ddf.map_partitions(
        _transform_partition,
        workflow_nodes,
        additional_columns=additional_columns,
        meta=meta,
        enforce_metadata=False,
    )


def _get_stat_ops(nodes):
    return set(node for node in iter_nodes(nodes) if isinstance(node.op, StatOperator))


def _get_schemaless_nodes(nodes):
    schemaless_nodes = []
    for node in iter_nodes(nodes):
        if node.input_schema is None:
            schemaless_nodes.append(node)

    return set(schemaless_nodes)


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())


def _transform_partition(root_df, workflow_nodes, additional_columns=None):
    """Transforms a single partition by appyling all operators in a WorkflowNode"""
    output = None

    for node in workflow_nodes:
        node_input_cols = _get_unique(node.input_schema.column_names)
        node_output_cols = _get_unique(node.output_schema.column_names)
        addl_input_cols = set(node.dependency_columns.names)

        # Build input dataframe
        if node.parents_with_dependencies:
            # If there are parents, collect their outputs
            # to build the current node's input
            input_df = None
            seen_columns = None

            for parent in node.parents_with_dependencies:
                parent_output_cols = _get_unique(parent.output_schema.column_names)
                parent_df = _transform_partition(root_df, [parent])
                if input_df is None or not len(input_df):
                    input_df = parent_df[parent_output_cols]
                    seen_columns = set(parent_output_cols)
                else:
                    new_columns = set(parent_output_cols) - seen_columns
                    input_df = _concat_columns([input_df, parent_df[list(new_columns)]])
                    seen_columns.update(new_columns)

            # Check for additional input columns that aren't generated by parents
            # and fetch them from the root dataframe
            unseen_columns = set(node.input_schema.column_names) - seen_columns
            addl_input_cols = addl_input_cols.union(unseen_columns)

            # TODO: Find a better way to remove dupes
            addl_input_cols = addl_input_cols - set(input_df.columns)

            if addl_input_cols:
                input_df = _concat_columns([input_df, root_df[list(addl_input_cols)]])
        else:
            # If there are no parents, this is an input node,
            # so pull columns directly from root df
            input_df = root_df[node_input_cols + list(addl_input_cols)]

        # Compute the node's output
        if node.op:
            try:
                # use input_columns to ensure correct grouping (subgroups)
                selection = node.input_columns.resolve(node.input_schema)
                output_df = node.op.transform(selection, input_df)
            except Exception:
                LOG.exception("Failed to transform operator %s", node.op)
                raise
            if output_df is None:
                raise RuntimeError("Operator %s didn't return a value during transform" % node.op)
        else:
            output_df = input_df

        # Combine output across node loop iterations

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        if output is None:
            output = output_df[node_output_cols]
        else:
            output = _concat_columns([output, output_df[node_output_cols]])

    if additional_columns:
        output = _concat_columns([output, root_df[_get_unique(additional_columns)]])

    return output
