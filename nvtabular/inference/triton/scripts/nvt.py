import numpy as np
import pandas as pd
import cupy as cp
#import cudf

class numpy_holder(object):
    pass

class nvtabular:

  def deserialize(self, path_config):
    self.stats = 2

  def transform(self, cont_array_info):

    cont_in_cai = dict()
    cont_in_cai['shape'] = (cont_array_info['shape_x'], cont_array_info['shape_y']) 
    cont_in_cai['data'] = (cont_array_info['data_in'], False) 
    cont_in_cai['strides'] = None
    cont_in_cai['descr'] = [('', '<f4')]
    cont_in_cai["typestr"] = "<f4"; 
    cont_in_cai["version"] = 3; 

    cont_out_cai = dict()
    cont_out_cai['shape'] = (cont_array_info['shape_x'], cont_array_info['shape_y']) 
    cont_out_cai['data'] = (cont_array_info['data_out'], False) 
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

    cont_in_cupy = cp.array(cont_in)
    cont_in_cupy = cont_in_cupy * self.stats
    cont_in_numpy = cp.asnumpy(cont_in_cupy)
    for i in range(len(cont_in_numpy)):
        cont_out[i] = cont_in_numpy[i]
    
    #df = cudf.DataFrame()
    #df['key'] = [0, 1, 2, 3, 4]
    #df['val'] = [float(i + 10) for i in range(5)] 
    #print(df)

    #df = pd.DataFrame(data=cont_in)
    #a = cudf.from_pandas(df)


    

   
