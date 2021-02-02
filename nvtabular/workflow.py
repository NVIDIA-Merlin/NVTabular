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
import json
import logging
import os
import sys
import time
import warnings
from typing import TYPE_CHECKING, Optional

import cloudpickle
import cudf
import dask
from dask.core import flatten

from nvtabular.column_group import ColumnGroup, iter_nodes
from nvtabular.io.dataset import Dataset
from nvtabular.ops import StatOperator
from nvtabular.worker import clean_worker_cache

LOG = logging.getLogger("nvtabular")


if TYPE_CHECKING:
    import distributed


class Workflow:
    """
    The Workflow class applies a graph of operations onto a dataset, letting you transform
    datasets to do feature engineering and preprocessing operations. This class follows an API
    similar to Transformers in sklearn: we first 'fit' the workflow by calculating statistics
    on the dataset, and then once fit we can 'transform' datasets by applying these statistics.

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
    column_group: ColumnGroup
        The graph of operators this workflow should apply
    client: distributed.Client, optional
        The Dask distributed client to use for multi-gpu processing and multi-node processing
    """

    def __init__(self, column_group: ColumnGroup, client: Optional["distributed.Client"] = None):
        self.column_group = column_group
        self.client = client
        self.input_dtypes = None
        self.output_dtypes = None

    def transform(self, dataset: Dataset) -> Dataset:
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

    def fit(self, dataset: Dataset):
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
        stat_ops = {op: _get_stat_ops(op.parents) for op in _get_stat_ops([self.column_group])}

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
        input_dtypes = dataset.to_ddf().dtypes
        self.input_dtypes = dict(zip(input_dtypes.index, input_dtypes))
        output_dtypes = self.transform(dataset).to_ddf().head(1).dtypes
        self.output_dtypes = dict(zip(output_dtypes.index, output_dtypes))

    def fit_transform(self, dataset: Dataset) -> Dataset:
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
        for stat in _get_stat_ops([self.column_group]):
            stat.op.set_storage_path(path, copy=True)

        # generate a file of all versions used to generate this bundle
        with open(os.path.join(path, "metadata.json"), "w") as o:
            json.dump(
                {
                    "versions": {
                        "nvtabular": nvt_version,
                        "cudf": cudf.__version__,
                        "python": sys.version,
                    },
                    "generated_timestamp": int(time.time()),
                },
                o,
            )

        # dump out the full workflow (graph/stats/operators etc) using cloudpickle
        with open(os.path.join(path, "workflow.pkl"), "wb") as o:
            cloudpickle.dump(self.column_group, o)

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
        versions = meta["versions"]
        check_version(versions["nvtabular"], nvt_version, "nvtabular")
        check_version(versions["cudf"], cudf.__version__, "cudf")
        check_version(versions["python"], sys.version, "python")

        # load up the workflow object di
        column_group = cloudpickle.load(open(os.path.join(path, "workflow.pkl"), "rb"))

        # we might have been copied since saving, update all the stat ops
        # with the new path to their storage locations
        for stat in _get_stat_ops([column_group]):
            stat.op.set_storage_path(path, copy=False)

        return cls(column_group, client)

    def clear_stats(self):
        for stat in _get_stat_ops([self.column_group]):
            stat.op.clear()

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

    # TODO: constructing meta like this loses dtype information on the ddf
    # sets it all to 'float64'. We should propogate dtype information along
    # with column names in the columngroup graph
    return ddf.map_partitions(
        _transform_partition,
        column_groups,
        meta=cudf.DataFrame({k: [] for k in columns}),
    )


def _get_stat_ops(nodes):
    return set(node for node in iter_nodes(nodes) if isinstance(node.op, StatOperator))


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
            if gdf is None:
                raise RuntimeError(
                    "Operator %s didn't return a value during transform" % column_group.op
                )

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        for column in column_group.flattened_columns:
            if column not in gdf:
                raise ValueError(
                    f"Failed to find {column} in output of {column_group}, which"
                    f" has columns {gdf.columns}"
                )
            output[column] = gdf[column]
    return output
