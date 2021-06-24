from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "movielens"
shape = (1, 1)

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    user_id_data = np.random.randint(1, 1000, shape).astype(np.int64)
    inputs = [
        httpclient.InferInput(
            "user_id", user_id_data.shape, np_to_triton_dtype(user_id_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(user_id_data)

    outputs = [
        httpclient.InferRequestedOutput("user_vector"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    print(
        "user_id: {}, user_vector: {}".format(
            user_id_data,
            response.as_numpy("user_vector"),
        )
    )
