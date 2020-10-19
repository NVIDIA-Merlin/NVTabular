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
import cudf
from nvtx import annotate

from .minmax import MinMax
from .moments import Moments
from .operator import CONT
from .transform_operator import DFOperator


class Normalize(DFOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the mean std method.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add Normalize to the workflow for continuous columns
        proc.add_cont_feature(nvt.ops.Normalize())

    Parameters
    ----------
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default False
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [Moments(columns=self.columns)]

    @annotate("Normalize_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names or not stats_context["stds"]:
            return
        gdf = self.apply_mean_std(gdf, stats_context, cont_names)
        return gdf

    def apply_mean_std(self, gdf, stats_context, cont_names):
        new_gdf = cudf.DataFrame()
        for name in cont_names:
            if stats_context["stds"][name] > 0:
                new_col = f"{name}_{self._id}"
                new_gdf[new_col] = (gdf[name] - stats_context["means"][name]) / (
                    stats_context["stds"][name]
                )
                new_gdf[new_col] = new_gdf[new_col].astype("float32")
        return new_gdf


class NormalizeMinMax(DFOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the min max method.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add NormalizeMinMax to the workflow for continuous columns
        proc.add_cont_feature(nvt.ops.NormalizeMinMax())

    Parameters
    ----------
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default False
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [MinMax(columns=self.columns)]

    @annotate("NormalizeMinMax_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names or not stats_context["mins"]:
            return
        gdf = self.apply_min_max(gdf, stats_context, cont_names)
        return gdf

    def apply_min_max(self, gdf, stats_context, cont_names):
        new_gdf = cudf.DataFrame()
        for name in cont_names:
            dif = stats_context["maxs"][name] - stats_context["mins"][name]
            new_col = f"{name}_{self._id}"
            if dif > 0:
                new_gdf[new_col] = (gdf[name] - stats_context["mins"][name]) / dif
            elif dif == 0:
                new_gdf[new_col] = gdf[name] / (2 * gdf[name])
            new_gdf[new_col] = new_gdf[new_col].astype("float32")
        return new_gdf
