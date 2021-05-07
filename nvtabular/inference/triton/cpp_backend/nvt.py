import numpy as np
import cudf
import nvtabular
from cudf.core.column import as_column, build_column
from cudf.utils.dtypes import is_list_dtype

class numpy_holder(object):
    pass

class TritonNVTabularModel:

  def initialize(self, path_workflow):
    self.workflow = nvtabular.Workflow.load(path_workflow)

    self.input_multihots = {
        col: dtype for col, dtype in self.workflow.input_dtypes.items() if is_list_dtype(dtype)
    }
    print("************ python initialize called")
    
  def transform(self, input_names, inputs, output_names):
    print("************ python transform called")
    input_df = cudf.DataFrame()
    
    for i in range(len(inputs)):
      inputs[i]['strides'] = None
      holder_in = numpy_holder()
      holder_in.__array_interface__ = inputs[i]
      np_in = np.array(holder_in, copy=False)
      input_df[input_names[i]] = np_in
    
    #output_df = self.workflow.transform(nvtabular.Dataset(input_df)).to_ddf().compute()
    output_df = nvtabular.workflow._transform_partition(
        input_df, [self.workflow.column_group]
    )
 
    output = {col:input_df[col].values_host for col in input_df.columns}
    return output
