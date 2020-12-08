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


def parse_args():
    parser = argparse.ArgumentParser(description=("Cloud Bash Script Generator"))

    # Number of GPUS
    parser.add_argument(
        "-d",
        "--devices",
        default="0",
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3").',
    )

    # Cloud provider
    parser.add_argument(
        "-c",
        "--cloud",
        choices=["aws", "gcp"],
        default="aws",
        type=str,
        help="Cloud provider to use (Default 'aws')",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Select dataset based on cloud option
    if args.cloud == "aws":
        dataset = "s3://merlin-datasets/crit_int_pq/"
    else:
        dataset = "gs://merlin-datasets/crit_int_pq/"

    # Programs to run within the container
    tests = "pytest /nvtabular/tests"
    benchmark = (
        "python /nvtabular/examples/dask-nvtabular-criteo-benchmark.py -d "
        + args.devices
        + " --data-path "
        + dataset
        + " --out-path /tmp/ --freq-limit 0 --part-mem-frac 0.16 --device-limit-frac 0.8"
        + " --device-pool-frac 0.9"
    )

    # Pull and run container
    docker = (
        "docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host "
        + '--cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.3 /bin/bash -c "source activate rapids '
        + tests
        + " && "
        + benchmark
    )

    # Generate Bash Script
    with open("run.sh", "w") as rsh:
        rsh.write("! /bin/bash\n")
        rsh.write(docker + "\n")


if __name__ == "__main__":
    main(parse_args())
