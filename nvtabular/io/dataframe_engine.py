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
try:
    import cudf
    import dask_cudf
except ImportError:
    cudf = None
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph

from .dataset_engine import DatasetEngine


class DataFrameDatasetEngine(DatasetEngine):
    """DataFrameDatasetEngine allows NVT to interact with a dask_cudf.DataFrame object
    in the same way as a dataset on disk.
    """

    def __init__(self, ddf, cpu=False, moved_collection=None):
        # we're not calling the constructor of the base class - since it has a bunch of
        # of parameters that assumes we're using files. TODO: refactor
        # pylint: disable=super-init-not-called
        self._ddf = ddf
        self.cpu = cpu
        self.moved_collection = moved_collection or False

    def to_ddf(self, columns=None, cpu=None):
        # Check if we are using cpu
        cpu = self.cpu if cpu is None else cpu

        # Move data from gpu to cpu if necessary
        _ddf = self._move_ddf("cpu") if (cpu and not self.cpu) else self._ddf

        if isinstance(columns, list):
            return _ddf[columns]
        elif isinstance(columns, str):
            return _ddf[[columns]]
        return _ddf

    def to_cpu(self):
        if self.cpu:
            return
        self._ddf = self._move_ddf("cpu")
        self.cpu = True
        self.moved_collection = not self.moved_collection

    def to_gpu(self):
        if not self.cpu:
            return
        self._ddf = self._move_ddf("gpu")
        self.cpu = False
        self.moved_collection = not self.moved_collection

    @property
    def num_rows(self):
        return len(self._ddf)

    def _move_ddf(self, destination):
        """Move the collection between cpu and gpu memory."""
        _ddf = self._ddf
        if (
            self.moved_collection
            and isinstance(_ddf.dask, HighLevelGraph)
            and hasattr(_ddf.dask, "key_dependencies")
        ):
            # If our collection has already been moved, and if the
            # underlying graph is a `HighLevelGraph`, we can just
            # drop the last "from_pandas-..." layer if the current
            # destination is "cpu", or we can drop the last
            # "to_pandas-..." layer if the destination is "gpu".
            search_name = "from_pandas-" if destination == "cpu" else "to_pandas-"

            pandas_conversion_layer = None
            pandas_conversion_dep = None
            for k, v in _ddf.dask.dependents.items():
                if k.startswith(search_name) and v == set():
                    pandas_conversion_layer = k
                    break
            if pandas_conversion_layer:
                deps = list(_ddf.dask.dependencies[pandas_conversion_layer])
                if len(deps) == 1:
                    pandas_conversion_dep = deps[0]

            if pandas_conversion_layer and pandas_conversion_dep:
                # We have met the criteria to remove the last "from/to_pandas-" layer
                new_layers = {
                    k: v for k, v in _ddf.dask.layers.items() if k != pandas_conversion_layer
                }
                new_deps = {
                    k: v for k, v in _ddf.dask.dependencies.items() if k != pandas_conversion_layer
                }
                hlg = HighLevelGraph(
                    layers=new_layers,
                    dependencies=new_deps,
                    key_dependencies=_ddf.dask.key_dependencies,
                )

                _meta = (
                    _ddf._meta.to_pandas() if destination == "cpu" else cudf.from_pandas(_ddf._meta)
                )
                return new_dd_object(hlg, pandas_conversion_dep, _meta, _ddf.divisions)

        if destination == "cpu":
            # Just extend the existing graph to move the collection to cpu
            return _ddf.to_dask_dataframe()

        elif destination == "gpu":
            # Just extend the existing graph to move the collection to gpu
            return dask_cudf.from_dask_dataframe(_ddf)

        else:
            raise ValueError(f"destination {destination} not recognized.")
