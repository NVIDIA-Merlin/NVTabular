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
from collections import defaultdict
from io import BytesIO

import cudf
import cupy
import dask_cudf
import numba.cuda as cuda
import numpy as np
import pyarrow.parquet as pq
from cudf._lib.nvtx import annotate
from cudf.io.parquet import ParquetWriter
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.dataframe.utils import group_split_dispatch
from dask.distributed import get_worker
from dask.utils import natural_sort_key, parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from pyarrow.compat import guid
from nvtabular.io import GPUDatasetIterator


class WriterCache:
    def __init__(self):
        self.pq_writer_cache = {}

    def __del__(self):
        for path, (pw, fpath) in self.pq_writer_cache.items():
            pw.close()

    def get_pq_writer(self, prefix, s, mem):
        pw, fil = self.pq_writer_cache.get(prefix, (None, None))
        if pw is None:
            if mem:
                fil = BytesIO()
                pw = ParquetWriter(fil, compression=None)
                self.pq_writer_cache[prefix] = (pw, fil)
            else:
                outfile_id = guid() + ".parquet"
                full_path = ".".join([prefix, outfile_id])
                pw = ParquetWriter(full_path, compression=None)
                self.pq_writer_cache[prefix] = (pw, full_path)
        return pw


def get_cache():
    try:
        worker = get_worker()
    except ValueError:
        # This is a metadata operation, so there is no "worker"
        # TODO: Handle metadata operations in a smarter way
        return None
    if not hasattr(worker, "pw_cache"):
        worker.pw_cache = WriterCache()
    return worker.pw_cache


def clean_pw_cache():
    worker = get_worker()
    if hasattr(worker, "pw_cache"):
        del worker.pw_cache
    return


def _write_metadata(meta_list):
    # TODO: Write _metadata file here (need to collect metadata)
    return meta_list


def _shuffle_gdf(gdf, gdf_size=None):
    gdf_size = gdf_size or len(gdf)
    arr = cupy.arange(gdf_size)
    cupy.random.shuffle(arr)
    return gdf.iloc[arr]


@annotate("write_output_partition", color="green", domain="nvt_python")
def _write_output_partition(gdf, processed_path, shuffle, nsplits, fs):
    gdf_size = len(gdf)
    if shuffle == "full":
        # Dont need a real sort if we are doing in memory later
        typ = np.min_scalar_type(nsplits * 2)
        ind = cupy.random.choice(cupy.arange(nsplits, dtype=typ), gdf_size)
        result = group_split_dispatch(gdf, ind, nsplits, ignore_index=True)
        del ind
        del gdf
        # Write each split to a separate file
        for s, df in result.items():
            prefix = fs.sep.join([processed_path, "split." + str(s)])
            pw = get_cache().get_pq_writer(prefix, s, mem=True)
            pw.write_table(df)
    else:
        # We should do a real sort here
        if shuffle == "partial":
            gdf = _shuffle_gdf(gdf, gdf_size=gdf_size)
        splits = list(range(0, gdf_size, int(gdf_size / nsplits)))
        if splits[-1] < gdf_size:
            splits.append(gdf_size)
        # Write each split to a separate file
        for s in range(0, len(splits) - 1):
            prefix = fs.sep.join([processed_path, "split." + str(s)])
            pw = get_cache().get_pq_writer(prefix, s, mem=False)
            pw.write_table(gdf.iloc[splits[s] : splits[s + 1]])
    return gdf_size  # TODO: Make this metadata


@annotate("worker_shuffle", color="green", domain="nvt_python")
def _worker_shuffle(processed_path, fs):
    paths = []
    for path, (pw, bio) in get_cache().pq_writer_cache.items():
        pw.close()

        gdf = cudf.io.read_parquet(bio, index=False)
        bio.close()

        gdf = _shuffle_gdf(gdf)
        rel_path = "shuffled.%s.parquet" % (guid())
        full_path = fs.sep.join([processed_path, rel_path])
        gdf.to_parquet(full_path, compression=None, index=False)
        paths.append(full_path)
    return paths


class DaskDataset:
    """ DaskDataset Class
        Converts a dataset into a dask_cudf DataFrame on demand

    Parameters
    -----------
    path : str or list of str
        Dataset path (or list of paths). If string, should specify
        a specific file or directory path. If this is a directory
        path, the directory structure must be flat (nested directories
        are not yet supported).
    engine : str or DatasetEngine
        DatasetEngine object or string identifier of engine. Current
        string options include: ("parquet").
    part_size : str or int
        Desired size (in bytes) of each Dask partition.
        If None, part_mem_fraction will be used to calculate the
        partition size.  Note that the underlying engine may allow
        other custom kwargs to override this argument.
    part_mem_fraction : float (default 0.125)
        Fractional size of desired dask partitions (relative
        to GPU memory capacity). Ignored if part_size is passed
        directly. Note that the underlying engine may allow other
        custom kwargs to override this argument.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    **kwargs :
        Other arguments to be passed to DatasetEngine.
    """

    def __init__(
        self,
        path,
        engine=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        **kwargs,
    ):
        self.kwargs = kwargs

        if part_size:
            # If a specific partition size is given, use it directly
            part_size = parse_bytes(part_size)
        else:
            # If a fractional partition size is given, calculate part_size
            part_mem_fraction = part_mem_fraction or 0.125
            assert part_mem_fraction > 0.0 and part_mem_fraction < 1.0
            if part_mem_fraction > 0.25:
                warnings.warn(
                    "Using very large partitions sizes for Dask. "
                    "Memory-related errors are likely."
                )
            part_size = int(cuda.current_context().get_memory_info()[1] * part_mem_fraction)

        # Engine-agnostic path handling
        if hasattr(path, "name"):
            path = stringify_path(path)
        storage_options = storage_options or {}
        fs, fs_token, paths = get_fs_token_paths(path, mode="rb", storage_options=storage_options)
        paths = sorted(paths, key=natural_sort_key)

        # If engine is not provided, try to infer from end of paths[0]
        if engine is None:
            engine = paths[0].split(".")[-1]

        self.itr = GPUDatasetIterator(paths, engine=engine)
        if isinstance(engine, str):
            if engine == "parquet":
                self.engine = ParquetDatasetEngine(paths, part_size, fs, fs_token, **kwargs)
            elif engine == "csv":
                self.engine = CSVDatasetEngine(paths, part_size, fs, fs_token, **kwargs)
            else:
                raise ValueError("Only parquet and csv supported (for now).")
        else:
            self.engine = engine(paths, part_size, fs, fs_token, **kwargs)

    def to_ddf(self, columns=None):
        return self.engine.to_ddf(columns=columns)
    
    def __iter__(self):
        for chunk in self.itr:
            yield chunk


class DatasetEngine:
    """ DatasetEngine Class

        Base class for Dask-powered IO engines. Engines must provide
        a ``to_ddf`` method.
    """

    def __init__(self, paths, part_size, fs, fs_token):
        self.paths = paths
        self.part_size = part_size
        self.fs = fs
        self.fs_token = fs_token

    def to_ddf(self, columns=None):
        raise NotImplementedError(""" Return a dask_cudf.DataFrame """)


class ParquetDatasetEngine(DatasetEngine):
    """ ParquetDatasetEngine

        Dask-based version of cudf.read_parquet.
    """

    def __init__(self, *args, row_groups_per_part=None):
        # TODO: Improve dask_cudf.read_parquet performance so that
        # this class can be slimmed down.
        super().__init__(*args)
        self._metadata, self._base = self.get_metadata()
        self._pieces = None
        if row_groups_per_part is None:
            # TODO: Use `total_byte_size` metadata if/when we figure out how to
            #       correct for apparent dict encoding of cat/string columns.
            path0 = self.fs.sep.join([self._base, self._metadata.row_group(0).column(0).file_path])
            rg_byte_size_0 = (
                cudf.io.read_parquet(path0, row_group=0).memory_usage(deep=True, index=True).sum()
            )
            self.row_groups_per_part = int(self.part_size / rg_byte_size_0)
        else:
            self.row_groups_per_part = int(row_groups_per_part)
        assert self.row_groups_per_part > 0

    @property
    def pieces(self):
        if self._pieces is None:
            self._pieces = self._get_pieces(self._metadata, self._base)
        return self._pieces

    def get_metadata(self):
        paths = self.paths
        fs = self.fs
        if len(paths) > 1:
            # This is a list of files
            dataset = pq.ParquetDataset(paths, filesystem=fs)
            base, fns = _analyze_paths(paths, fs)
        elif fs.isdir(paths[0]):
            # This is a directory
            dataset = pq.ParquetDataset(paths[0], filesystem=fs)
            allpaths = fs.glob(paths[0] + fs.sep + "*")
            base, fns = _analyze_paths(allpaths, fs)
        else:
            # This is a single file
            dataset = pq.ParquetDataset(paths[0], filesystem=fs)
            base = paths[0]
            fns = [None]

        metadata = None
        if dataset.metadata:
            # We have a metadata file
            return dataset.metadata, base
        else:
            # Collect proper metadata manually
            metadata = None
            for piece, fn in zip(dataset.pieces, fns):
                md = piece.get_metadata()
                if fn:
                    md.set_file_path(fn)
                if metadata:
                    metadata.append_row_groups(md)
                else:
                    metadata = md
            return metadata, base

    @annotate("get_pieces", color="green", domain="nvt_python")
    def _get_pieces(self, metadata, data_path):
        # get the number of row groups per file
        file_row_groups = defaultdict(int)
        for rg in range(metadata.num_row_groups):
            fpath = metadata.row_group(rg).column(0).file_path
            if fpath is None:
                raise ValueError("metadata is missing file_path string.")
            file_row_groups[fpath] += 1
        # create pieces from each file, limiting the number of row_groups in each piece
        pieces = []
        for filename, row_group_count in file_row_groups.items():
            row_groups = range(row_group_count)
            for i in range(0, row_group_count, self.row_groups_per_part):
                rg_list = list(row_groups[i : i + self.row_groups_per_part])
                full_path = self.fs.sep.join([data_path, filename])
                pieces.append((full_path, rg_list))
        return pieces

    @staticmethod
    @annotate("read_piece", color="green", domain="nvt_python")
    def read_piece(piece, columns):
        path, row_groups = piece
        return cudf.io.read_parquet(
            path,
            row_group=row_groups[0],
            row_group_count=len(row_groups),
            columns=columns,
            index=False,
        )

    def meta_empty(self, columns=None):
        path, _ = self.pieces[0]
        return cudf.io.read_parquet(path, row_group=0, columns=columns, index=False).iloc[:0]

    def to_ddf(self, columns=None):
        pieces = self.pieces
        name = "parquet-to-ddf-" + tokenize(self.fs_token, pieces, columns)
        dsk = {
            (name, p): (ParquetDatasetEngine.read_piece, piece, columns)
            for p, piece in enumerate(pieces)
        }
        meta = self.meta_empty(columns=columns)
        divisions = [None] * (len(pieces) + 1)
        return new_dd_object(dsk, name, meta, divisions)


class CSVDatasetEngine(DatasetEngine):
    """ CSVDatasetEngine

        Thin wrapper around dask_cudf.read_csv.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._meta = {}
        self.names = kwargs.pop("names", None)
        self.csv_kwargs = kwargs
        # CSV reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

    def to_ddf(self, columns=None):
        return dask_cudf.read_csv(
            self.paths, names=self.names, chunksize=self.part_size, **self.csv_kwargs
        )[columns]
