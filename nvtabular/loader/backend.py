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

from merlin.dtypes.shape import Shape
from merlin.schema import ColumnSchema, Tags


def _augment_schema(
    schema,
    cats=None,
    conts=None,
    labels=None,
    padded_cols=None,
    padded_lengths=None,
    pad=False,
    batch_size=0,
):
    labels = [labels] if isinstance(labels, str) else labels
    for label in labels or []:
        schema[label] = schema[label].with_tags(Tags.TARGET)
    for label in cats or []:
        schema[label] = schema[label].with_tags(Tags.CATEGORICAL)
    for label in conts or []:
        schema[label] = schema[label].with_tags(Tags.CONTINUOUS)

    for col in padded_cols or []:
        cs = schema[col]
        dims = Shape(((1, batch_size), None))

        if not cs.shape.dims[1].is_unknown:
            dims = dims.with_dim(1, cs.shape.dims[1])

        if pad:
            dims = dims.with_dim_min(1, padded_lengths[col])
        if padded_lengths and col in padded_lengths:
            dims = dims.with_dim_max(1, padded_lengths[col])

        schema[col] = ColumnSchema(
            name=cs.name, tags=cs.tags, dtype=cs.dtype, properties=cs.properties, dims=dims
        )

    return schema
