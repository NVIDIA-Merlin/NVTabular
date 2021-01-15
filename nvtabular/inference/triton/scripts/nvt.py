import numpy as np

class numpy_holder(object):
    pass

class nvtabular:

  def deserialize(self, path_config):
    print("deserialize called")

  def transform(self, cont_in_cai, cont_out_cai, cat):
    print("transform")

    cont_in_cai['strides'] = None
    cont_in_cai['descr'] = [('', '<f4')]
    cont_in_cai["typestr"] = "<f4"; 
    cont_in_cai["version"] = 3; 

    cont_out_cai['strides'] = None
    cont_out_cai['descr'] = [('', '<f4')]
    cont_out_cai["typestr"] = "<f4"; 
    cont_out_cai["version"] = 3; 

    holder_cont_in = numpy_holder()
    holder_cont_in.__array_interface__ = cont_in_cai
    cont_in = np.array(holder_cont_in, copy=False)

    holder_cont_out = numpy_holder()
    holder_cont_out.__array_interface__ = cont_out_cai
    cont_out = np.array(holder_cont_out, copy=False)

    for i in range(len(cont_in)):
        cont_out[i] = cont_in[i] * 2

    print(cont_in)
    print(cont_out)

   
