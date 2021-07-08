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
import numba.cuda
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from nvtabular.dispatch import DataFrameType, _build_cudf_list_column, _is_cpu_object, annotate

from .operator import ColumnNames, Operator


class ListSlice(Operator):
    """Slices a list column

    This operator provides the ability to slice list column by row. For example, to truncate a
    list column to only include the first 10 elements per row::

        truncated = column_names >> ops.ListSlice(10)

    Take the first 10 items, ignoring the first element::

        truncated = column_names >> ops.ListSlice(1, 11)

    Take the last 10 items from each row::

        truncated = column_names >> ops.ListSlice(-10)
    """

    def __init__(self, start, end=None):
        super().__init__()
        self.start = start
        self.end = end

        if self.start > 0 and self.end is None:
            self.end = self.start
            self.start = 0

        if self.end is None:
            self.end = np.iinfo(np.int64).max

    @annotate("ListSlice_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        on_cpu = _is_cpu_object(df)
        ret = type(df)()
        for col in columns:
            # handle CPU via normal python slicing (not very efficient)
            if on_cpu:
                ret[col] = [row[self.start : self.end] for row in df[col]]
            else:
                # figure out the size of each row from the list offsets
                c = df[col]._column
                offsets = c.offsets.values
                elements = c.elements.values

                # figure out the size of each row after slicing start/end
                new_offsets = cp.zeros(offsets.size, dtype=offsets.dtype)
                threads = 32
                blocks = (offsets.size + threads - 1) // threads

                # calculate new row offsets after slicing
                _calculate_row_sizes[blocks, threads](self.start, self.end, offsets, new_offsets)
                new_offsets = cp.cumsum(new_offsets).astype(offsets.dtype)

                # create a new array for the sliced elements
                new_elements = cp.zeros(new_offsets[-1].item(), dtype=elements.dtype)
                if new_elements.size:
                    _slice_rows[blocks, threads](
                        self.start, offsets, elements, new_offsets, new_elements
                    )

                # build up a list column with the sliced values
                ret[col] = _build_cudf_list_column(new_elements, new_offsets)

        return ret

    transform.__doc__ = Operator.transform.__doc__


@numba.cuda.jit
def _calculate_row_sizes(start, end, offsets, row_sizes):
    """given a slice (start/end) and existing offsets indicating row lengths, this
    calculates the size for each new row after slicing"""
    rowid = numba.cuda.grid(1)
    if rowid < offsets.size - 1:
        original_row_size = offsets[rowid + 1] - offsets[rowid]

        # handle negative slicing appropiately
        if start < 0:
            start = original_row_size + start
        if end < 0:
            end = original_row_size + end

        # clamp start/end to be in (0, original_row_size)
        start = min(max(0, start), original_row_size)
        end = min(max(0, end), original_row_size)

        row_sizes[rowid + 1] = end - start


@numba.cuda.jit
def _slice_rows(start, offsets, elements, new_offsets, new_elements):
    """slices rows of a list column. requires the 'new_offsets' to
    be previously calculated (meaning that we don't need the 'end' slice index
    since thats baked into the new_offsets"""
    rowid = numba.cuda.grid(1)
    if rowid < (new_offsets.size - 1):
        if start >= 0:
            offset = offsets[rowid] + start
        else:
            offset = offsets[rowid + 1] + start
            if offset < offsets[rowid]:
                offset = offsets[rowid]

        for new_offset in range(new_offsets[rowid], new_offsets[rowid + 1]):
            new_elements[new_offset] = elements[offset]
            offset += 1
