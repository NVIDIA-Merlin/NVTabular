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
import collections

import cudf
import dask
import pandas as pd
import pyarrow as pa
from dask.base import tokenize
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from nvtx import annotate

from nvtabular.dispatch import _to_arrow
from nvtabular.worker import clean_worker_cache, get_worker_cache

from .shuffle import Shuffle
from .writer_factory import _writer_cls_factory, writer_factory


@annotate("write_output_partition", color="green", domain="nvt_python")
def _write_output_partition(
    df,
    processed_path,
    shuffle,
    out_files_per_proc,
    fs,
    cat_names,
    cont_names,
    label_names,
    output_format,
    num_threads,
    cpu,
):
    df_size = len(df)
    out_files_per_proc = out_files_per_proc or 1

    # Get cached writer (or create/cache a new one)
    with get_worker_cache("writer") as writer_cache:
        writer = writer_cache.get(processed_path, None)
        if writer is None:
            writer = writer_factory(
                output_format,
                processed_path,
                out_files_per_proc,
                shuffle,
                use_guid=True,
                bytes_io=(shuffle == Shuffle.PER_WORKER),
                num_threads=num_threads,
                cpu=cpu,
            )
            writer.set_col_names(labels=label_names, cats=cat_names, conts=cont_names)
            writer_cache[processed_path] = writer

        # Add data
        writer.add_data(df)

    return df_size


@annotate("write_table_list", color="green", domain="nvt_python")
def _write_table_list(
    tables,
    fn,
    output_path,
    shuffle,
    fs,
    cat_names,
    cont_names,
    label_names,
    output_format,
    num_threads,
    cpu,
):

    writer = writer_factory(
        output_format,
        output_path,
        1,
        shuffle,
        bytes_io=(shuffle == Shuffle.PER_WORKER),
        num_threads=num_threads,
        cpu=cpu,
        fns=[fn],
    )
    writer.set_col_names(labels=label_names, cats=cat_names, conts=cont_names)

    # Add data
    num_rows = 0
    for table in tables:
        if isinstance(table, pa.Table):
            if cpu:
                writer.add_data(table.to_pandas())
            else:
                writer.add_data(cudf.DataFrame.from_arrow(table))
        else:
            writer.add_data(table)
        num_rows += len(table)

    # Return metadata and row-count in dict
    return writer.close()


def _write_metadata_files(md_list, output_path, output_format, cpu):

    # Separate and merge metadata
    general_md = []
    special_md = []
    for md in md_list:
        general_md.append(md[0])
        special_md.append(md[1])
    general_md = _merge_general_metadata(general_md)
    special_md = dict(collections.ChainMap(*special_md))

    # Write metadata files
    if not isinstance(output_path, str):
        output_path = str(output_path)
    wc, fs = _writer_cls_factory(output_format, output_path, cpu)
    wc.write_general_metadata(general_md, fs, output_path)
    wc.write_special_metadata(special_md, fs, output_path)


def _ddf_to_dataset(
    ddf,
    fs,
    output_path,
    shuffle,
    file_partition_map,
    out_files_per_proc,
    cat_names,
    cont_names,
    label_names,
    output_format,
    client,
    num_threads,
    cpu,
):

    # Construct graph for Dask-based dataset write
    token = tokenize(ddf, shuffle, out_files_per_proc, cat_names, cont_names, label_names)
    name = "write-processed-" + token
    write_name = name + "-partition" + token

    # Check that the data is in the correct place
    assert isinstance(ddf._meta, pd.DataFrame) is cpu

    dsk = {}
    task_list = []
    if file_partition_map is not None:
        # Use specified mapping of data to output files
        cached_writers = False
        ddf2 = ddf
        if not cpu and len(file_partition_map) < ddf.npartitions:
            # Aggregate partitions in arrow format (in host memory)
            ddf2 = ddf.map_partitions(_to_arrow, meta=ddf._meta)

        for fn, parts in file_partition_map.items():
            task_list.append((write_name, fn))
            dsk[task_list[-1]] = (
                _write_table_list,
                [(ddf2._name, part) for part in parts],
                fn,
                output_path,
                shuffle,
                fs,
                cat_names,
                cont_names,
                label_names,
                output_format,
                num_threads,
                cpu,
            )
        dsk[name] = (
            _write_metadata_files,
            task_list,
            output_path,
            output_format,
            cpu,
        )
    else:
        cached_writers = True
        ddf2 = ddf
        for idx in range(ddf.npartitions):
            key = (write_name, idx)
            dsk[key] = (
                _write_output_partition,
                (ddf._name, idx),
                output_path,
                shuffle,
                out_files_per_proc,
                fs,
                cat_names,
                cont_names,
                label_names,
                output_format,
                num_threads,
                cpu,
            )
            task_list.append(key)
        dsk[name] = (lambda x: x, task_list)

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf2])
    out = Delayed(name, graph)

    # Trigger write execution
    if client:
        out = client.compute(out).result()
    else:
        out = dask.compute(out, scheduler="synchronous")[0]

    if cached_writers:
        # Follow-up Shuffling and _metadata creation
        _finish_dataset(client, ddf, output_path, fs, output_format, cpu)


def _finish_dataset(client, ddf, output_path, fs, output_format, cpu):
    # Finish data writing
    if client:
        client.cancel(ddf)
        ddf = None
        out = client.run(_worker_finish, output_path)

        general_md = []
        special_md = []
        for (gen, spec) in out.values():
            general_md.append(gen)
            if spec:
                special_md.append(spec)

        general_md = _merge_general_metadata(general_md)
        special_md = dict(collections.ChainMap(*special_md))
    else:
        ddf = None
        general_md, special_md = _worker_finish(output_path)

    # Write metadata on client
    if not isinstance(output_path, str):
        output_path = str(output_path)

    wc, fs = _writer_cls_factory(output_format, output_path, cpu)
    wc.write_general_metadata(general_md, fs, output_path)
    wc.write_special_metadata(special_md, fs, output_path)

    # Clean writer caches
    if client:
        client.run(clean_worker_cache, "writer")
    else:
        clean_worker_cache("writer")


def _worker_finish(processed_path):
    general_md, special_md = {}, {}
    with get_worker_cache("writer") as writer_cache:
        writer = writer_cache.get(processed_path, None)
        if writer:
            general_md, special_md = writer.close()

    return general_md, special_md


def _merge_general_metadata(meta_list):
    """Combine list of "general" metadata dicts into
    a single dict
    """
    if not meta_list:
        return {}
    meta = None
    for md in meta_list:
        if meta:
            if "data_paths" in md:
                meta["data_paths"] += md["data_paths"]
            if "file_stats" in md:
                meta["file_stats"] += md["file_stats"]
        else:
            meta = md.copy()
            if "data_paths" not in meta:
                meta["data_paths"] = []
            if "file_stats" not in meta:
                meta["file_stats"] = []
    return meta
