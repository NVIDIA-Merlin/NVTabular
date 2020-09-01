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

from numba import cuda


def device_mem_size(kind="total"):
    if kind not in ["free", "total"]:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        import pynvml

        pynvml.nvmlInit()
        if kind == "free":
            warnings.warn("get_memory_info is not supported. Using total device memory from NVML.")
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total)
        pynvml.nvmlShutdown()
    return size
