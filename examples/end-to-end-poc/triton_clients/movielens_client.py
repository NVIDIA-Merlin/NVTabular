from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "movielens"
shape = (1, 1)

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    user_id_data = np.random.randint(1, 100000, shape).astype(np.int32)
    inputs = [
        httpclient.InferInput(
            "user_id", user_id_data.shape, np_to_triton_dtype(user_id_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(user_id_data)

    outputs = [
        httpclient.InferRequestedOutput("filtered_candidate_ids"),
        # httpclient.InferRequestedOutput("candidate_distances")
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()

    print(
        "user_id:\n{}\ncandidate_ids:\n{}\nlength:\n{}\n".format(
            user_id_data,
            response.as_numpy("filtered_candidate_ids"),
            len(response.as_numpy("filtered_candidate_ids").T),
        )
    )
