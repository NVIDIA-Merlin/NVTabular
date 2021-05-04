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
import warnings

import cudf
import uavro as ua
from dask.base import tokenize
from dask.dataframe.core import new_dd_object

from .dataset_engine import DatasetEngine


class AvroDatasetEngine(DatasetEngine):
    """AvroDatasetEngine

    Uses `uavro` to decompose dataset into groups of avro blocks.
    Uses `cudf` to create new partitions.
    """

    def __init__(self, paths, part_size, storage_options=None, cpu=False, **kwargs):
        super().__init__(paths, part_size, storage_options=storage_options, cpu=cpu)
        if kwargs != {}:
            raise ValueError("Unexpected AvroDatasetEngine argument(s).")
        self.blocksize = part_size

        # Avro reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

        if self.cpu:
            raise ValueError("cpu=True not supported for AvroDatasetEngine.")

    def to_ddf(self, columns=None, cpu=None):

        # Check if we are using cpu
        cpu = self.cpu if cpu is None else cpu
        if cpu:
            raise ValueError("cpu=True not supported for AvroDatasetEngine.")

        # Get list of pieces for each output
        pieces, meta = self.process_metadata(columns=columns)

        # TODO: Remove warning and avoid use of uavro in read_partition when
        # cudf#6529 is fixed (https://github.com/rapidsai/cudf/issues/6529)
        if len(pieces) > len(self.paths):
            warnings.warn(
                "Row-subset selection in cudf avro reader is currently broken. "
                "Using uavro engine until cudf#6529 is addressed. "
                "EXPECT POOR PERFORMANCE!! (compared to cuio-based reader)"
            )

        # Construct collection
        token = tokenize(self.fs, self.paths, self.part_size, columns)
        read_avro_name = "read-avro-partition-" + token
        dsk = {
            (read_avro_name, i): (AvroDatasetEngine.read_partition, self.fs, piece, columns)
            for i, piece in enumerate(pieces)
        }
        return new_dd_object(dsk, read_avro_name, meta.iloc[:0], [None] * (len(pieces) + 1))

    def to_cpu(self):
        raise ValueError("cpu=True not supported for AvroDatasetEngine.")

    def to_gpu(self):
        self.cpu = False

    def process_metadata(self, columns=None):

        with open(self.paths[0], "rb") as fo:
            header = ua.core.read_header(fo)

            # Use first block for metadata
            num_rows = header["blocks"][0]["nrows"]
            file_byte_count = header["blocks"][0]["size"]
            meta = cudf.io.read_avro(self.paths[0], skiprows=0, num_rows=num_rows)

            # Convert the desired in-memory GPU size to the expected
            # on-disk storage size (blocksize)
            df_byte_count = meta.memory_usage(deep=True).sum()
            self.blocksize = int(float(file_byte_count) / df_byte_count * self.part_size)

        # Break apart files at the "Avro block" granularity
        pieces = []
        for path in self.paths:
            file_size = self.fs.du(path)
            if file_size > self.blocksize:
                part_count = 0
                with open(path, "rb") as fo:
                    header = ua.core.read_header(fo)
                    ua.core.scan_blocks(fo, header, file_size)
                    blocks = header["blocks"]

                    file_row_offset, part_row_count = 0, 0
                    file_block_offset, part_block_count = 0, 0
                    file_byte_offset, part_byte_count = blocks[0]["offset"], 0

                    for i, block in enumerate(blocks):
                        part_row_count += block["nrows"]
                        part_block_count += 1
                        part_byte_count += block["size"]
                        if part_byte_count >= self.blocksize:
                            pieces.append(
                                {
                                    "path": path,
                                    "rows": (file_row_offset, part_row_count),
                                    "blocks": (file_block_offset, part_block_count),
                                    "bytes": (file_byte_offset, part_byte_count),
                                }
                            )
                            part_count += 1
                            file_row_offset += part_row_count
                            file_block_offset += part_block_count
                            file_byte_offset += part_byte_count
                            part_row_count = part_block_count = part_byte_count = 0

                    if part_block_count:
                        pieces.append(
                            {
                                "path": path,
                                "rows": (file_row_offset, part_row_count),
                                "blocks": (file_block_offset, part_block_count),
                                "bytes": (file_byte_offset, part_byte_count),
                            }
                        )
                        part_count += 1
                if part_count == 1:
                    # No need to specify a byte range since we
                    # will need to read the entire file anyway.
                    pieces[-1] = {"path": pieces[-1]["path"]}
            else:
                pieces.append({"path": path})

        return pieces, meta

    @classmethod
    def read_partition(cls, fs, piece, columns):

        path = piece["path"]
        if "rows" in piece:

            # See: (https://github.com/rapidsai/cudf/issues/6529)
            # Using `uavro` library for now. This means we must covert
            # data to pandas, and then to cudf (which is much slower
            # than `cudf.read_avro`). TODO: Once `num_rows` is fixed,
            # this can be changed to:
            #
            #   skiprows, num_rows = piece["rows"]
            #   df = cudf.io.read_avro(
            #       path, skiprows=skiprows, num_rows=num_rows
            #   )

            block_offset, part_blocks = piece["blocks"]
            file_size = fs.du(piece["path"])
            with fs.open(piece["path"], "rb") as fo:
                header = ua.core.read_header(fo)
                ua.core.scan_blocks(fo, header, file_size)
                header["blocks"] = header["blocks"][block_offset : block_offset + part_blocks]

                # Adjust the total row count
                nrows = 0
                for block in header["blocks"]:
                    nrows += block["nrows"]
                header["nrows"] = nrows

                # Read in as pandas and convert to cudf (avoid block scan)
                df = cudf.from_pandas(
                    ua.core.filelike_to_dataframe(fo, file_size, header, scan=False)
                )
        else:
            df = cudf.io.read_avro(path)

        # Deal with column selection
        if columns is None:
            columns = list(df.columns)
        return df[columns]
