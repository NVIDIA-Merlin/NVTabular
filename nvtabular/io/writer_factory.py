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
from fsspec.core import get_fs_token_paths

from .hugectr import HugeCTRWriter
from .parquet import CPUParquetWriter, GPUParquetWriter


def writer_factory(
    output_format,
    output_path,
    out_files_per_proc,
    shuffle,
    use_guid=False,
    bytes_io=False,
    num_threads=0,
    cpu=False,
    fns=None,
    suffix=None,
):
    if output_format is None:
        return None

    writer_cls, fs = _writer_cls_factory(output_format, output_path, cpu=cpu)
    return writer_cls(
        output_path,
        num_out_files=out_files_per_proc,
        shuffle=shuffle,
        fs=fs,
        use_guid=use_guid,
        bytes_io=bytes_io,
        num_threads=num_threads,
        cpu=cpu,
        fns=fns,
        suffix=suffix,
    )


def _writer_cls_factory(output_format, output_path, cpu=None):
    if output_format == "parquet" and cpu:
        writer_cls = CPUParquetWriter
    elif output_format == "parquet":
        writer_cls = GPUParquetWriter
    elif output_format == "hugectr":
        writer_cls = HugeCTRWriter
    else:
        raise ValueError("Output format not yet supported.")

    fs = get_fs_token_paths(output_path)[0]
    return writer_cls, fs
