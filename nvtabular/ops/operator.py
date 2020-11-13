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
import warnings

CONT = "continuous"
CAT = "categorical"
ALL = "all"


class Operator:
    """
    Base class for all operator classes.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self._id_set = None

    @property
    def _id(self):
        c_id = self._id_set
        if not self._id_set:
            c_id = str(self.__class__.__name__)
        return c_id

    def _set_id(self, id_to_set):
        # only set one time
        if not self._id_set:
            self._id_set = id_to_set
        else:
            warnings.warn(
                f"trying to reset operator id, for operator: {self._id_set} to {id_to_set}"
            )

    def describe(self):
        raise NotImplementedError("All operators must have a desription.")

    def get_columns(self, cols_ctx, cols_grp, target_cols):
        # providing any operator with direct list of columns overwrites cols dict
        # burden on user to ensure columns exist in dataset (as discussed)
        if self.columns:
            return self.columns
        tar_cols = []
        for tar in target_cols:
            if tar in cols_ctx[cols_grp].keys():
                tar_cols = tar_cols + cols_ctx[cols_grp][tar]
        if len(tar_cols) < 1:
            tar_cols = cols_ctx[cols_grp]["base"]
        return tar_cols
