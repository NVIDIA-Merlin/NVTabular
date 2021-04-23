import sys
import cudf
import numpy as np
import tritonclient.grpc as httpclient
import tritonhttpclient
from tritonclient.utils import *

import nvtabular as nvt

DIR = "/model/"
DATA_DIR = DIR + "data/"

# This test assumes that the Triton Inference Server has been started already
# using this command:
# tritonserver --model-repository=/model/models/ --backend-config=hugectr,test_model=/model/models/test_model/1/model.json --backend-config=hugectr,supportlonglong=true --model-control-mode=explicit --load-model=test_model_ens


def test_nvt_hugectr_inference():

    try:
        triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
        print("client created.")
    except Exception as e:
        print("Channel creation failed: " + str(e))

    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")

    model_name = "test_model_ens"
    col_names = ["userId", "movieId"]
    # read in a batch of data to get transforms for
    batch = cudf.read_parquet(DATA_DIR + "valid/*.parquet", num_rows=64)[col_names]

    # convert the batch to a triton inputs
    columns = [(col, batch[col]) for col in col_names]
    inputs = []

    col_dtypes = [np.int64, np.int64]
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(col_dtypes[i])
        d = d.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(name, d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d)

    # placeholder variables for the output
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("OUTPUT0"))
    # make the request
    with httpclient.InferenceServerClient("localhost:8001") as client:
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
