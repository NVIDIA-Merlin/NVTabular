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
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph

from .dataset_engine import DatasetEngine


def _gddf_to_ddf(_gddf):
    # Generate a dask.dataframe.DataFrame from a dask_cudf.Frame

    # Check if our dask_cudf collection was just created from a
    # dask.dataframe collection.  If so, we can just drop the
    # cpu -> gpu "from_pandas-..." layer in the graph
    if isinstance(_gddf.dask, HighLevelGraph) and hasattr(_gddf.dask, "key_dependencies"):
        from_pandas_layer = None
        from_pandas_dep = None
        for k, v in _gddf.dask.dependents.items():
            if k.startswith("from_pandas-") and v == set():
                from_pandas_layer = k
                break
        if from_pandas_layer:
            deps = [d for d in _gddf.dask.dependencies[from_pandas_layer]]
            if len(deps) == 1:
                from_pandas_dep = deps[0]

        if from_pandas_layer and from_pandas_dep:
            # We have met the criteria to remove the last "from_pandas-" layer
            new_layers = {k: v for k, v in _gddf.dask.layers.items() if k != from_pandas_layer}
            new_deps = {k: v for k, v in _gddf.dask.dependencies.items() if k != from_pandas_layer}
            hlg = HighLevelGraph(
                layers=new_layers,
                dependencies=new_deps,
                key_dependencies=_gddf.dask.key_dependencies,
            )
            return new_dd_object(hlg, from_pandas_dep, _gddf._meta.to_pandas(), _gddf.divisions)
        else:
            return _gddf.to_dask_dataframe()
    else:
        # Just extend the existing graph to move the collection to cpu
        return _gddf.to_dask_dataframe()


def _ddf_to_gddf(_ddf):
    import dask_cudf

    return dask_cudf.from_dask_dataframe(_ddf)


class DataFrameDatasetEngine(DatasetEngine):
    """DataFrameDatasetEngine allows NVT to interact with a dask_cudf.DataFrame object
    in the same way as a dataset on disk.
    """

    def __init__(self, ddf, cpu=False):
        self._ddf = ddf
        self.cpu = cpu

    def to_ddf(self, columns=None, cpu=None):
        # Check if we are using cpu
        cpu = self.cpu if cpu is None else cpu

        # Move data from gpu to cpu if necessary
        _ddf = _gddf_to_ddf(self._ddf) if (cpu and not self.cpu) else self._ddf

        if isinstance(columns, list):
            return _ddf[columns]
        elif isinstance(columns, str):
            return _ddf[[columns]]
        return _ddf

    def to_cpu(self):
        if self.cpu:
            return
        self._ddf = _gddf_to_ddf(self._ddf)

    def to_gpu(self):
        if not self.cpu:
            return
        self._ddf = _ddf_to_gddf(self._ddf)

    @property
    def num_rows(self):
        return len(self._ddf)
