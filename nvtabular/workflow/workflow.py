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
import pandas as pd

from merlin.dag import Graph
from merlin.dag.executors import DaskExecutor
from merlin.io import Dataset
from merlin.schema import Schema
from nvtabular.ops import StatOperator
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
        workflow.fit(merlin.io.Dataset(TRAIN_PATH))

        # transform the training and validation datasets and write out as parquet
        workflow.transform(merlin.io.Dataset(TRAIN_PATH)).to_parquet(output_path=TRAIN_OUT_PATH)
        workflow.transform(merlin.io.Dataset(VALID_PATH)).to_parquet(output_path=VALID_OUT_PATH)

    Parameters
    ----------
    output_node: WorkflowNode
        The last node in the graph of operators this workflow should apply
    """

    def __init__(self, output_node: WorkflowNode, client: Optional["distributed.Client"] = None):
        self.graph = Graph(output_node)
        self.executor = DaskExecutor(client)

    def transform(self, dataset: Dataset) -> Dataset:
        """Transforms the dataset by applying the graph of operators to it. Requires the ``fit``
        method to have already been called, or calculated statistics to be loaded from disk

        This method returns a Dataset object, with the transformations lazily loaded. None
        of the actual computation will happen until the produced Dataset is consumed, or
        written out to disk.

        Parameters
        -----------
        dataset: Dataset
            Input dataset to transform

        Returns
        -------
        Dataset
            Transformed Dataset with the workflow graph applied to it
        """
        return self._transform_impl(dataset)

    def fit_schema(self, input_schema: Schema):
        """Fits the schema onto the workflow, computing the Schema for each node in the Workflow Graph

        Parameters
        ----------
        input_schema : Schema
            The input schema to use

        Returns
        -------
        Workflow
            This workflow where each node in the graph has a fitted schema
        """
        self.graph.construct_schema(input_schema)
        return self

    @property
    def input_dtypes(self):
        return self.graph.input_dtypes

    @property
    def input_schema(self):
        return self.graph.input_schema

    @property
    def output_schema(self):
        return self.graph.output_schema

    @property
    def output_dtypes(self):
        return self.graph.output_dtypes

    @property
    def output_node(self):
        return self.graph.output_node

    def _input_columns(self):
        return self.graph._input_columns()

    def remove_inputs(self, input_cols) -> "Workflow":
        """Removes input columns from the workflow.

        This is useful for the case of inference where you might need to remove label columns
        from the processed set.

        Parameters
        ----------
        input_cols : list of str
            List of column names to

        Returns
        -------
        Workflow
            This workflow with the input columns removed from it

        See Also
        --------
        merlin.dag.Graph.remove_inputs
        """
        self.graph.remove_inputs(input_cols)
        return self

    def fit(self, dataset: Dataset) -> "Workflow":
        """Calculates statistics for this workflow on the input dataset

        Parameters
        -----------
        dataset: Dataset
            The input dataset to calculate statistics for. If there is a train/test split this
            data should be the training dataset only.

        Returns
        -------
        Workflow
            This Workflow with statistics calculated on it
        """
        self.clear_stats()

        if not self.graph.output_schema:
            self.graph.construct_schema(dataset.schema)

        ddf = dataset.to_ddf(columns=self._input_columns())

        # Get a dictionary mapping all StatOperators we need to fit to a set of any dependent
        # StatOperators (having StatOperators that depend on the output of other StatOperators
        # means that will have multiple phases in the fit cycle here)
        stat_op_nodes = {
            node: Graph.get_nodes_by_op_type(node.parents_with_dependencies, StatOperator)
            for node in Graph.get_nodes_by_op_type([self.graph.output_node], StatOperator)
        }

        while stat_op_nodes:
            # get all the StatOperators that we can currently call fit on (no outstanding
            # dependencies)
            current_phase = [
                node for node, dependencies in stat_op_nodes.items() if not dependencies
            ]
            if not current_phase:
                # this shouldn't happen, but lets not infinite loop just in case
                raise RuntimeError("failed to find dependency-free StatOperator to fit")

            self.executor.fit(ddf, current_phase)

            # Remove all the operators we processed in this phase, and remove
            # from the dependencies of other ops too
            for node in current_phase:
                stat_op_nodes.pop(node)
            for dependencies in stat_op_nodes.values():
                dependencies.difference_update(current_phase)

        # This captures the output dtypes of operators like LambdaOp where
        # the dtype can't be determined without running the transform
        self._transform_impl(dataset, capture_dtypes=True).sample_dtypes()
        self.graph.construct_schema(dataset.schema, preserve_dtypes=True)

        return self

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Convenience method to both fit the workflow and transform the dataset in a single
        call. Equivalent to calling ``workflow.fit(dataset)`` followed by
        ``workflow.transform(dataset)``

        Parameters
        -----------
        dataset: Dataset
            Input dataset to calculate statistics on, and transform results

        Returns
        -------
        Dataset
            Transformed Dataset with the workflow graph applied to it

        See Also
        --------
        fit
        transform
        """
        self.fit(dataset)
        return self.transform(dataset)

    def _transform_impl(self, dataset: Dataset, capture_dtypes=False):
        if not self.graph.output_schema:
            self.graph.construct_schema(dataset.schema)

        ddf = dataset.to_ddf(columns=self._input_columns())

        return Dataset(
            self.executor.transform(
                ddf, self.output_node, self.output_dtypes, capture_dtypes=capture_dtypes
            ),
            cpu=dataset.cpu,
            base_dataset=dataset.base_dataset,
            schema=self.output_schema,
        )

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
        for stat in Graph.get_nodes_by_op_type([self.output_node], StatOperator):
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
    def load(cls, path, client=None) -> "Workflow":
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
            The Workflow loaded from disk
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
        for stat in Graph.get_nodes_by_op_type([workflow.output_node], StatOperator):
            stat.op.set_storage_path(path, copy=False)

        return workflow

    def clear_stats(self):
        """Removes calculated statistics from each node in the workflow graph

        See Also
        --------
        nvtabular.ops.stat_operator.StatOperator.clear
        """
        for stat in Graph.get_nodes_by_op_type([self.graph.output_node], StatOperator):
            stat.op.clear()
