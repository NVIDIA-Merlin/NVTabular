#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cudf
import dask
from dask.core import flatten

from nvtabular.column_group import ColumnGroup, iter_nodes
from nvtabular.io.dataset import Dataset
from nvtabular.ops import StatOperator
from nvtabular.worker import clean_worker_cache


LOG = logging.getLogger("nvtabular")


class Workflow:
    """
    Workflow organizes and runs all feature engineering and preprocessing operators for your
    workflow.

    Parameters
    ----------
    columns_group: ColumnGroup
        The graph of transformations this workflow should apply
    client: Dask.client, optional
        The Dask client to use for multi-gpu processing
    """

    def __init__(self, column_group: ColumnGroup, client=None):
        self.column_group = column_group
        self.client = client

    def transform(self, dataset):
        """Transforms the dataset by applying the graph of operators to it. Requires the 'fit'
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
        ddf = dataset.to_ddf(columns=self._input_columns())
        return Dataset(_transform_ddf(ddf, self.column_group), client=self.client)

    def fit(self, dataset):
        """Calculates statistics for this workflow on the input dataset

        Parameters
        -----------
        dataset: Dataset
            The input dataset to calculate statistics for. If there is a train/test split this
            data should be the training dataset only.
        """
        self._clear_worker_cache()
        ddf = dataset.to_ddf(columns=self._input_columns())

        # Get a dictionary mapping all StatOperators we need to fit to a set of any dependant
        # StatOperators (having StatOperators that depend on the output of other StatOperators
        # means that will have multiple phases in the fit cycle here)
        def get_stat_ops(nodes):
            return set(node for node in iter_nodes(nodes) if isinstance(node.op, StatOperator))

        stat_ops = {op: get_stat_ops(op.parents) for op in get_stat_ops([self.column_group])}

        while stat_ops:
            # get all the StatOperators that we can currently call fit on (no outstanding
            # dependencies)
            current_phase = [op for op, dependencies in stat_ops.items() if not dependencies]
            if not current_phase:
                # this shouldn't happen, but lets not infinite loop just in case
                raise RuntimeError("failed to find dependency-free StatOperator to fit")

            stats, ops = [], []
            for column_group in current_phase:
                # apply transforms necessary for the inputs to the current column group, ignoring
                # the transforms from the statop itself
                transformed_ddf = _transform_ddf(ddf, column_group.parents)

                op = column_group.op
                try:
                    stats.append(op.fit(column_group.input_column_names, transformed_ddf))
                    ops.append(op)
                except Exception:
                    LOG.exception("Failed to fit operator %s", column_group.op)
                    raise

            if self.client:
                results = [r.result() for r in self.client.compute(stats)]
            else:
                results = dask.compute(stats, schedule="synchronous")[0]

            for computed_stats, op in zip(results, ops):
                op.fit_finalize(computed_stats)

            # Remove all the operators we processed in this phase, and remove
            # from the dependencies of other ops too
            for stat_op in current_phase:
                stat_ops.pop(stat_op)
            for dependencies in stat_ops.values():
                dependencies.difference_update(current_phase)

    def fit_transform(self, dataset):
        """Convenience method to both fit the workflow and transform the dataset in a single
        call. Equivalent to calling workflow.fit(dataset) followed by workflow.transform(dataset)

        Parameters
        -----------
        dataset: Dataset

        Returns
        -------
        Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

    def _input_columns(self):
        input_nodes = set(node for node in iter_nodes([self.column_group]) if not node.parents)
        return list(set(col for node in input_nodes for col in node.flattened_columns))

    def _clear_worker_cache(self):
        # Clear worker caches to be "safe"
        if self.client:
            self.client.run(clean_worker_cache)
        else:
            clean_worker_cache()


def _transform_ddf(ddf, column_groups):
    if isinstance(column_groups, ColumnGroup):
        column_groups = [column_groups]

    columns = list(flatten(cg.flattened_columns for cg in column_groups))

    return ddf.map_partitions(
        lambda gdf: _transform_partition(gdf, column_groups),
        meta=cudf.DataFrame({k: [] for k in columns}),
    )


def _transform_partition(root_gdf, column_groups):
    """ Transforms a single partition by appyling all operators in a ColumnGroup """
    output = cudf.DataFrame()
    for column_group in column_groups:
        # collect dependencies recursively if we have parents
        if column_group.parents:
            gdf = cudf.DataFrame()
            for parent in column_group.parents:
                parent_gdf = _transform_partition(root_gdf, [parent])
                for column in parent.flattened_columns:
                    gdf[column] = parent_gdf[column]
        else:
            # otherwise select the input from the root gdf
            gdf = root_gdf[column_group.flattened_columns]

        # apply the operator if necessary
        if column_group.op:
            try:
                gdf = column_group.op.transform(column_group.input_column_names, gdf)
            except Exception:
                LOG.exception("Failed to transform operator %s", column_group.op)
                raise

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        for column in column_group.flattened_columns:
            output[column] = gdf[column]
    return output
