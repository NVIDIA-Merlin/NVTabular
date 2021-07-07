from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "movielens_session_filter"

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    candidate_id_data = np.random.randint(1, 1000, (100, 1)).astype(np.int32)
    session_id_data = np.random.randint(1, 1000, (100, 1)).astype(np.int32)

    inputs = [
        httpclient.InferInput(
            "candidate_movie_ids", candidate_id_data.shape, np_to_triton_dtype(
                candidate_id_data.dtype)
        ),
        httpclient.InferInput(
            "session_movie_ids", session_id_data.shape, np_to_triton_dtype(session_id_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(candidate_id_data)
    inputs[1].set_data_from_numpy(session_id_data)

    outputs = [
        httpclient.InferRequestedOutput("filtered_movie_ids"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    print(
        "candidate_ids: {}\n session_ids: {}\n filtered_ids: {}".format(
            candidate_id_data,
            session_id_data,
            response.as_numpy("filtered_movie_ids"),
        )
    )
