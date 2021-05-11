import cudf
import numpy as np
from cudf.core.column import as_column, build_column
from cudf.utils.dtypes import is_list_dtype

import nvtabular


class numpy_holder(object):
    pass


class TritonNVTabularModel:
    def initialize(self, path_workflow, dtypes):
        self.workflow = nvtabular.Workflow.load(path_workflow)
        self.input_multihots = {
            col: dtype for col, dtype in self.workflow.input_dtypes.items() if is_list_dtype(dtype)
        }
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
            self.output_dtypes[name] = self._convertToNumpyDtype(dtype)

        self.lenghts = list()


    def transform(self, input_names, inputs, output_names):
        input_df = cudf.DataFrame()

        for i in range(len(inputs)):
            inputs[i]["strides"] = None
            holder_in = numpy_holder()
            holder_in.__array_interface__ = inputs[i]
            np_in = np.array(holder_in, copy=False)
            input_df[input_names[i]] = np_in

        output_df = nvtabular.workflow._transform_partition(input_df, [self.workflow.column_group])
        output = dict()
        self.lenghts = list()
        for name in output_df.columns:
            col = output_df[name]
            if is_list_dtype(col.dtype):
                # convert list values to match TF dataloader
                values = col.list.leaves.values_host.astype(
                    self.output_dtypes[name + "__values"]
                )
                output[name + "__values"] = values
                self.lenghts.append(len(output[name + "__values"]))

                offsets = col._column.offsets.values_host.astype(
                   self.output_dtypes[name + "__nnzs"]
                )
                nnzs = offsets[1:] - offsets[:-1]
                nnzs = nnzs.reshape(len(nnzs), 1)
                output[name + "__nnzs"] = nnzs
                self.lenghts.append(len(output[name + "__nnzs"]))
            else:
                output[name] = col.values_host.astype(self.output_dtypes[name])
                self.lenghts.append(len(output[name]))
        
        return output

    def get_lengths(self):
        return self.lenghts

    def _convertToNumpyDtype(self, dtype):
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
            return np.float54
        else:
            return np.bytes
