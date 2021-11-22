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

import io
from threading import Thread

import numpy as np
from pyarrow import parquet as pq

try:
    import cudf
    from cudf.core.column import as_column, build_categorical_column
except ImportError:
    cudf = None

#
# Parquet-Specific Utilities
#


def _optimized_read_partition_remote(
    fs, pieces, columns, index, categories=(), partitions=(), **kwargs
):
    # This is a specialized version of `CudfEngine.read_partition`
    # for remote filesystems. This implementation is intended to
    # replace the upstream `read_partition` classmethod until
    # remote-filesystem handling is optimized in cudf/dask-cudf

    if columns is not None:
        columns = list(columns)
    if isinstance(index, list):
        columns += index

    # Check that this is a single-piece read on a non-local filesystem
    if not isinstance(pieces, list):
        pieces = [pieces]
    if len(pieces) > 1:
        raise ValueError(
            "The `_custom_read_partition` code path is not designed to "
            "handle a multi-element `pieces` argument."
        )
    if cudf.utils.ioutils._is_local_filesystem(fs):
        raise ValueError(
            "The `_custom_read_partition` code path is not intended "
            "for use on local filesystems."
        )

    # Unpack contents of the single piece
    if isinstance(pieces[0], str):
        path = pieces[0]
        row_group = None
        partition_keys = []
    else:
        (path, row_group, partition_keys) = pieces[0]

    # Call optimized read utility
    df = _optimized_read_remote(path, row_group, columns, fs, **kwargs)

    #
    # Code below is directly copied from cudf-21.08
    #

    if index and (index[0] in df.columns):
        df = df.set_index(index[0])
    elif index is False and set(df.index.names).issubset(columns):
        # If index=False, we need to make sure all of the
        # names in `columns` are actually in `df.columns`
        df.reset_index(inplace=True)

    if partition_keys:
        if partitions is None:
            raise ValueError("Must pass partition sets")
        for i, (name, index2) in enumerate(partition_keys):
            categories = [val.as_py() for val in partitions.levels[i].dictionary]

            col = as_column(index2).as_frame().repeat(len(df))._data[None]
            df[name] = build_categorical_column(
                categories=categories,
                codes=as_column(col.base_data, dtype=col.dtype),
                size=col.size,
                offset=col.offset,
                ordered=False,
            )

    return df


def _optimized_read_remote(path, row_groups, columns, fs, **kwargs):

    if row_groups is not None and not isinstance(row_groups, list):
        row_groups = [row_groups]

    # Get byte-ranges that are known to contain the
    # required data for this read
    byte_ranges, footer, file_size = _get_parquet_byte_ranges(
        path, row_groups, columns, fs, **kwargs
    )

    # Call cudf.read_parquet on the dummy buffer
    strings_to_cats = kwargs.get("strings_to_categorical", False)
    return cudf.read_parquet(
        # Wrap in BytesIO since cudf will sometimes use
        # pyarrow to parse the metadata (and pyarrow
        # cannot read from a bytes object)
        io.BytesIO(
            # Transfer the required bytes with fsspec
            _fsspec_data_transfer(
                path,
                fs,
                byte_ranges=byte_ranges,
                footer=footer,
                file_size=file_size,
                add_par1_magic=True,
                **kwargs,
            )
        ),
        engine="cudf",
        columns=columns,
        row_groups=row_groups,
        strings_to_categorical=strings_to_cats,
        **kwargs.get("read", {}),
    )


def _get_parquet_byte_ranges(
    path,
    rgs,
    columns,
    fs,
    bytes_per_thread=256_000_000,
    **kwargs,
):
    # The purpose of this utility is to return a list
    # of byte ranges (in path) that are known to contain
    # the data needed to read `columns` and `rgs`

    # Step 0 - Get size of file
    file_size = fs.size(path)

    # Return early if the file is too small to merit
    # optimized data transfer
    if file_size <= bytes_per_thread:
        return None, None, file_size

    # Step 1 - Get 32 KB from tail of file.
    #
    # This "sample size" can be tunable, but should
    # always be >= 8 bytes (so we can read the footer size)
    tail_size = 32_000
    footer_sample = fs.tail(path, tail_size)

    # Step 2 - Read the footer size and re-read a larger
    #          tail if necessary
    footer_size = int.from_bytes(footer_sample[-8:-4], "little")
    if tail_size < (footer_size + 8):
        footer_sample = fs.tail(path, footer_size + 8)

    # Step 3 - Collect required byte ranges
    byte_ranges = []
    md = pq.ParquetFile(io.BytesIO(footer_sample)).metadata
    for r in range(md.num_row_groups):
        # Skip this row-group if we are targeting
        # specific row-groups
        if rgs is None or r in rgs:
            row_group = md.row_group(r)
            for c in range(row_group.num_columns):
                column = row_group.column(c)
                name = column.path_in_schema
                # Skip this column if we are targeting
                # specific columns, and this name is not
                # in the list.
                #
                # Note that `column.path_in_schema` may
                # modify the column name for list and struct
                # columns. For example, a column named "a"
                # may become "a.list.element"
                split_name = name.split(".")[0]
                if columns is None or name in columns or split_name in columns:
                    file_offset0 = column.dictionary_page_offset
                    if file_offset0 is None:
                        file_offset0 = column.data_page_offset
                    num_bytes = column.total_compressed_size
                    byte_ranges.append((file_offset0, num_bytes))

    return byte_ranges, footer_sample, file_size


#
# General Fsspec Data-transfer Optimization Code
#


def _fsspec_data_transfer(
    path_or_fob,
    fs,
    byte_ranges=None,
    footer=None,
    file_size=None,
    add_par1_magic=None,
    bytes_per_thread=256_000_000,
    max_gap=64_000,
    mode="rb",
    **kwargs,
):

    # Calculate total file size
    file_size = file_size or fs.size(path_or_fob)

    # Check if a direct read makes the most sense
    if not byte_ranges and bytes_per_thread >= file_size:
        return fs.open(path_or_fob, mode=mode, cache_type="none").read()

    # Threaded read into "dummy" buffer
    buf = np.zeros(file_size, dtype="b")
    if byte_ranges:

        # Optimize/merge the ranges
        byte_ranges = _merge_ranges(
            byte_ranges,
            max_block=bytes_per_thread,
            max_gap=max_gap,
        )

        # Call multi-threaded data transfer of
        # remote byte-ranges to local buffer
        _read_byte_ranges(
            path_or_fob,
            byte_ranges,
            buf,
            fs,
            **kwargs,
        )

        # Add Header & Footer bytes
        if footer is not None:
            footer_size = len(footer)
            buf[-footer_size:] = np.frombuffer(footer[-footer_size:], dtype="b")

        # Add parquet magic bytes (optional)
        if add_par1_magic:
            buf[:4] = np.frombuffer(b"PAR1", dtype="b")
            if footer is None:
                buf[-4:] = np.frombuffer(b"PAR1", dtype="b")

    else:
        byte_ranges = [
            (b, min(bytes_per_thread, file_size - b)) for b in range(0, file_size, bytes_per_thread)
        ]
        _read_byte_ranges(
            path_or_fob,
            byte_ranges,
            buf,
            fs,
            **kwargs,
        )

    return buf.tobytes()


def _merge_ranges(byte_ranges, max_block=256_000_000, max_gap=64_000):
    # Simple utility to merge small/adjacent byte ranges
    new_ranges = []
    if not byte_ranges:
        # Early return
        return new_ranges

    offset, size = byte_ranges[0]
    for (new_offset, new_size) in byte_ranges[1:]:
        gap = new_offset - (offset + size)
        if gap > max_gap or (size + new_size + gap) > max_block:
            # Gap is too large or total read is too large
            new_ranges.append((offset, size))
            offset = new_offset
            size = new_size
            continue
        size += new_size + gap
    new_ranges.append((offset, size))
    return new_ranges


def _assign_block(fs, path_or_fob, local_buffer, offset, nbytes):
    with fs.open(path_or_fob, mode="rb", cache_type="none") as fob:
        fob.seek(offset)
        local_buffer[offset : offset + nbytes] = np.frombuffer(
            fob.read(nbytes),
            dtype="b",
        )


def _read_byte_ranges(
    path_or_fob,
    ranges,
    local_buffer,
    fs,
    **kwargs,
):

    workers = []
    for (offset, nbytes) in ranges:
        if len(ranges) > 1:
            workers.append(
                Thread(target=_assign_block, args=(fs, path_or_fob, local_buffer, offset, nbytes))
            )
            workers[-1].start()
        else:
            _assign_block(fs, path_or_fob, local_buffer, offset, nbytes)

    for worker in workers:
        worker.join()
