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

from merlin.schema import ColumnSchema, Tags


def _augment_schema(
    schema,
    cats=None,
    conts=None,
    labels=None,
    sparse_names=None,
    sparse_max=None,
    sparse_as_dense=False,
):
    labels = [labels] if isinstance(labels, str) else labels
    for label in labels or []:
        schema[label] = schema[label].with_tags(Tags.TARGET)
    for label in cats or []:
        schema[label] = schema[label].with_tags(Tags.CATEGORICAL)
    for label in conts or []:
        schema[label] = schema[label].with_tags(Tags.CONTINUOUS)

    # Set the appropriate properties for the sparse_names/sparse_max/sparse_as_dense
    for col in sparse_names or []:
        cs = schema[col]
        properties = cs.properties
        if sparse_max and col in sparse_max:
            properties["value_count"] = {"max": sparse_max[col]}
        schema[col] = ColumnSchema(
            name=cs.name,
            tags=cs.tags,
            dtype=cs.dtype,
            is_list=True,
            is_ragged=not sparse_as_dense,
            properties=properties,
        )

    return schema
