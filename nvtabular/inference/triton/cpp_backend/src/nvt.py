import numpy as np
import cudf
import nvtabular

class numpy_holder(object):
    pass

class nvt:

  def deserialize(self, path_workflow):
    self.workflow = nvtabular.Workflow.load(path_workflow)

  def transform(self, input_names, inputs):
    input_df = cudf.DataFrame()
    
    for i in range(len(inputs)):
      inputs[i]['strides'] = None
      holder_in = numpy_holder()
      holder_in.__array_interface__ = inputs[i]
      np_in = np.array(holder_in, copy=False)
      input_df[input_names[i]] = np_in
    
    output_df = self.workflow.transform(nvtabular.Dataset(input_df)).to_ddf().compute()
    output = {col:output_df[col].values_host for col in output_df.columns}
    return output
