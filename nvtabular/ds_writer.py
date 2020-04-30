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

import glob
import os

import cudf
import numpy as np
import pyarrow.parquet as pq

try:
    import cupy as cp
except ImportError:
    import numpy as cp


class FileIterator:
    def __init__(self, path, nfiles, shuffle=True, **kwargs):
        self.path = path
        self.nfiles = nfiles
        self.shuffle = shuffle
        self.ind = 0
        self.inds = np.arange(self.nfiles)
        if self.shuffle:
            np.random.shuffle(self.inds)

    def __iter__(self):
        self.ind = 0
        self.inds = np.arange(self.nfiles)
        if self.shuffle:
            np.random.shuffle(self.inds)
        return self

    def __next__(self):
        if self.ind >= self.nfiles:
            raise StopIteration
        self.ind += 1
        # if self.name, return that naming convention.
        return "%s/ds_part.%d.parquet" % (self.path, self.ind - 1)


class DatasetWriter:
    def __init__(self, path, nfiles=1, **kwargs):
        self.path = path
        self.nfiles = nfiles
        self.writers = {fn: None for fn in FileIterator(path, nfiles)}
        self.shared_meta_path = str(path) + "/_metadata"
        self.metadata = None
        self.new_metadata = {fn: [] for fn in FileIterator(path, nfiles)}

        # Check for _metadata
        metafile = glob.glob(self.shared_meta_path)
        if metafile:
            self.metadata = pq.ParquetDataset(metafile[0]).metadata

    def write(self, gdf, shuffle=True):

        # Shuffle the dataframe
        gdf_size = len(gdf)
        if shuffle:
            sort_key = "__sort_index__"
            arr = cp.arange(gdf_size)
            cp.random.shuffle(arr)
            gdf[sort_key] = cudf.Series(arr)
            gdf = gdf.sort_values(sort_key).drop(columns=[sort_key])

        # Write to
        chunk_size = int(gdf_size / self.nfiles)
        for i, fn in enumerate(FileIterator(self.path, self.nfiles)):
            s1 = i * chunk_size
            s2 = (i + 1) * chunk_size
            if i == (self.nfiles - 1):
                s2 = gdf_size
            chunk = gdf[s1:s2]
            pa_table = chunk.to_arrow()
            if self.writers[fn] is None:
                self.writers[fn] = pq.ParquetWriter(
                    fn, pa_table.schema, metadata_collector=self.new_metadata[fn],
                )
            self.writers[fn].write_table(pa_table)

    def write_metadata(self):
        self.close_writers()  # Writers must be closed to get metadata
        fns = [fn for fn in FileIterator(self.path, self.nfiles, shuffle=False)]
        if self.metadata is not None:
            _meta = self.metadata
            i_start = 0
        else:
            _meta = self.new_metadata[fns[0]]
            if _meta:
                _meta = _meta[0]
            i_start = 1
        for i in range(i_start, len(fns)):
            _meta_new = self.new_metadata[fns[i]]
            if _meta_new:
                _meta.append_row_groups(_meta_new[0])
        with open(self.shared_meta_path, "wb") as fil:
            _meta.write_metadata_file(fil)
        self.metadata = _meta
        return

    def close_writers(self):
        for fn, writer in self.writers.items():
            if writer is not None:
                writer.close()
                # Set row-group file paths
                self.new_metadata[fn][0].set_file_path(os.path.basename(fn))
                writer = None

    def __del__(self):
        self.close_writers()
