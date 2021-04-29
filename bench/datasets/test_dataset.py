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

import argparse
import json
import os

from tools.nvt_etl import nvt_etl

from nvtabular import Workflow


def main(args):
    # Get cats/conts/labels
    with open(args.config_file) as f:
        data = json.load(f)
    cats = data["cats"]
    conts = data["conts"]
    labels = data["labels"]

    # Perform ETL
    if not args.workflow_path:
        workflow = nvt_etl(
            args.data_path,
            args.out_path,
            args.devices,
            args.protocol,
            args.device_limit_frac,
            args.device_pool_frac,
            args.part_mem_frac,
            cats,
            conts,
            labels,
            args.out_files_per_proc,
        )
    else:
        workflow = Workflow.load(os.path.join(args.workflow_path, "workflow"))

    # Perform training
    # HugeCTR
    if args.target_framework == "hugectr":
        from tools.train_hugectr import train_hugectr

        train_hugectr(workflow, args.devices, args.out_path)
    # TensorFlow
    elif args.target_framework == "tensorflow":
        from tools.train_tensorflow import train_tensorflow

        train_tensorflow(workflow, args.out_path + "output", cats, conts, labels, 64 * 1024)
    # PyTorch
    else:
        from tools.train_pytorch import train_pytorch

        train_pytorch(workflow, args.out_path + "output", cats, conts, labels, 400000, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Test", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data-path", type=str, help="Input dataset path", required=True)

    parser.add_argument(
        "--out-path", type=str, help="Directory path to write output", required=True
    )

    parser.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
        "The number of visible devices dictates the number of Dask workers (GPU processes) "
        "The CUDA_VISIBLE_DEVICES environment variable will be used by default",
    )

    parser.add_argument(
        "--target-framework",
        choices=["hugectr", "tensorflow", "pytorch"],
        default="hugectr",
        type=str,
        help="Target Framework",
        required=True,
    )

    parser.add_argument(
        "--config-file",
        type=str,
        help="Configuration file",
        default="./configs/default_config.json",
        required=True,
    )

    parser.add_argument(
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use",
    )

    parser.add_argument(
        "--device-limit-frac",
        default=0.8,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity. "
        "The worker will try to spill data to host memory beyond this limit",
    )

    parser.add_argument(
        "--device-pool-frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity. "
        "If 0 is specified, the RMM pool will be disabled",
    )

    parser.add_argument(
        "--part-mem-frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )

    parser.add_argument(
        "--out-files-per-proc",
        default=8,
        type=int,
        help="Number of output files to write on each worker",
    )

    parser.add_argument("--workflow-path", type=str, help="Worflow path, so ETL is not performed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
