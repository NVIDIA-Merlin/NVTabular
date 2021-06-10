# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cudf
import numpy as np
from cudf.utils.dtypes import is_list_dtype

import nvtabular


# Empty class that is needed by the numpy array interface
class numpy_holder(object):
    pass


class TritonNVTabularModel:
    """TritonNVTabularModel for nvtabular C++ backend"""

    def initialize(self, path_workflow, dtypes):        
        self.workflow = nvtabular.Workflow.load(path_workflow)
        self.input_dtypes = {
            col: dtype
            for col, dtype in self.workflow.input_dtypes.items()
            if not is_list_dtype(dtype)
        }
        self.input_multihots = {
            col: dtype for col, dtype in self.workflow.input_dtypes.items() if is_list_dtype(dtype)
        }

        self.output_dtypes = dict()
        for name, dtype in dtypes.items():
            self.output_dtypes[name] = self._convert_to_numpy_dtype(dtype)
            

    def transform(self, input_names, inputs, output_names):
        """Transforms the inputs by running through a NVTabular workflow.transform
        function.
        """

        # Create numpy arrays using only the pointers. Then, create
        # cudf using the numpy arrays.
        input_df = cudf.DataFrame()
        for i in range(len(inputs)):
            inputs[i]["strides"] = None
            holder_in = numpy_holder()
            holder_in.__array_interface__ = inputs[i]
            np_in = np.array(holder_in, copy=False)
            input_df[input_names[i]] = np_in

        # Applies nvtabular transforms
        output_df = nvtabular.workflow._transform_partition(input_df, [self.workflow.column_group])
        output = dict()
        lengths = list()

        # Applies nvtabular transforms
        for name in output_df.columns:
            col = output_df[name]
            if is_list_dtype(col.dtype):
                # convert list values to match TF dataloader
                output[name + "__values"] = col.list.leaves.values_host.astype(self.output_dtypes[name + "__values"])
                lengths.append(len(output[name + "__values"]))

                offsets = col._column.offsets.values_host.astype(
                    self.output_dtypes[name + "__nnzs"]
                )
                nnzs = offsets[1:] - offsets[:-1]
                output[name + "__nnzs"] = nnzs
                lengths.append(len(output[name + "__nnzs"]))
            else:
                output[name] = col.values_host.astype(self.output_dtypes[name])
                lengths.append(len(output[name]))

        return (output, lengths)
    

    def _convert_to_numpy_dtype(self, dtype):
        if dtype == "invalid":
            raise Exception("wrong data type")
        elif dtype == "np.bool_":
            return np.bool_
        elif dtype == "np.uint8":
            return np.uint8
        elif dtype == "np.uint16":
            return np.uint16
        elif dtype == "np.uint32":
            return np.uint32
        elif dtype == "np.uint64":
            return np.uint64
        elif dtype == "np.int8":
            return np.int8
        elif dtype == "np.int16":
            return np.int16
        elif dtype == "np.int32":
            return np.int32
        elif dtype == "np.int64":
            return np.int64
        elif dtype == "np.float16":
            return np.float16
        elif dtype == "np.float32":
            return np.float32
        elif dtype == "np.float64":
            return np.float64
        else:
            return np.bytes
