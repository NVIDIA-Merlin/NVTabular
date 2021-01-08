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

import argparse
import json

import fsspec

import nvtabular as nvt


def parse_args():
    parser = argparse.ArgumentParser(description=("Dataset Inspect Tool"))
    # Config file
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Dataset columns type (Required)",
    )
    # Dataset path
    parser.add_argument(
        "-d",
        "--data_path",
        default="0",
        type=str,
        help="Input dataset path (Required)",
    )
    # Dataset format
    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        type=str,
        help="Dataset format (Default 'parquet')",
    )
    # Output file name
    parser.add_argument(
        "-o",
        "--output_file",
        default="dataset_info.json",
        type=str,
        help="Output file name (Default 'dataset_info.json')",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Get dataset columns
    with fsspec.open(args.config_file) as f:
        config = json.load(f)

    a = nvt.tools.DatasetInspector()
    a.inspect(args.data_path, args.format, config, args.output_file)


if __name__ == "__main__":
    main(parse_args())