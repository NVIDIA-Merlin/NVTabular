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
from nvtabular.dispatch import DataFrameType

from .operator import ColumnSelector, Operator


class AddTags(Operator):
    """
    This operator will add user defined tags to a Schema.
    """

    def __init__(self, tags=None):
        super().__init__()
        self.tags = tags or []

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    @property
    def output_tags(self):
        return self.tags


class AddMetadata(AddTags):
    """
    This operator will add user defined tags and properties
    to a Schema.
    """

    def __init__(self, tags=None, properties=None):
        super().__init__(tags)
        self.properties = properties or {}

    @property
    def output_properties(self):
        return self.properties


# Wrappers for common features
class TagAsUserID(Operator):
    @property
    def output_tags(self):
        return ["UserID"]


class TagAsItemID(AddTags):
    @property
    def output_tags(self):
        return ["ItemID"]


class TagAsUserFeatures(AddTags):
    @property
    def output_tags(self):
        return ["UserFeatures"]


class TagAsItemFeatures(AddTags):
    @property
    def output_tags(self):
        return ["ItemFeatures"]
