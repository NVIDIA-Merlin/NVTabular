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
from merlin.core.dispatch import DataFrameType
from merlin.schema.tags import Tags

from .operator import ColumnSelector, Operator


class AddMetadata(Operator):
    """
    This operator will add user defined tags and properties
    to a Schema.
    """

    def __init__(self, tags=None, properties=None):
        super().__init__()
        self.tags = tags or []
        self.properties = properties or {}

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    @property
    def output_tags(self):
        return self.tags

    @property
    def output_properties(self):
        return self.properties


class AddTags(AddMetadata):
    def __init__(self, tags=None):
        super().__init__(tags=tags)


class AddProperties(AddMetadata):
    def __init__(self, properties=None):
        super().__init__(properties=properties)


# Wrappers for common features
class TagAsUserID(AddTags):
    def __init__(self, tags=None):
        super().__init__(tags=[Tags.USER_ID, Tags.USER])


class TagAsItemID(AddTags):
    def __init__(self, tags=None):
        super().__init__(tags=[Tags.ITEM_ID, Tags.ITEM])


class TagAsUserFeatures(AddTags):
    def __init__(self, tags=None):
        super().__init__(tags=[Tags.USER])


class TagAsItemFeatures(AddTags):
    def __init__(self, tags=None):
        super().__init__(tags=[Tags.ITEM])
