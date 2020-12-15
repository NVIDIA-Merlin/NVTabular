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
import fsspec
import json
import nvtabular as nvt


def parse_args():
    parser = argparse.ArgumentParser(description=("Dataset Inspect Tool"))

    # Dataset path
    parser.add_argument(
        "-d",
        "--data_path",
        default="0",
        type=str,
        help='Input dataset path (Required)',
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
    # Get dataset
    dataset = nvt.Dataset(args.data_path, engine=args.format)
    ddf = dataset.to_ddf()
    
    # Dictionary to store collected information
    data = {}
    #data['num_rows'] = dataset.num_rows

    # Get columns and dtypes
    for col in ddf.columns:
        data[col] = {}
        data[col]['dtype'] = str(ddf[col].dtype)
        data[col]['min'] = ddf[col].min().compute()
        data[col]['max'] = ddf[col].max().compute()
    
    print(data)

    # Write json file
    with fsspec.open(args.output_file, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    main(parse_args())