from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "movielens_user_features"
shape = (1, 1)

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    user_id_data = np.random.randint(1, 1000, shape).astype(np.int32)
    inputs = [
        httpclient.InferInput(
            "user_id", user_id_data.shape, np_to_triton_dtype(user_id_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(user_id_data)

    outputs = [
        httpclient.InferRequestedOutput("movie_id_count"),
        httpclient.InferRequestedOutput("movie_ids__values"),
        httpclient.InferRequestedOutput("movie_ids__nnzs"),
        httpclient.InferRequestedOutput("genres__values"),
        httpclient.InferRequestedOutput("genres__nnzs"),
        httpclient.InferRequestedOutput("search_terms__values"),
        httpclient.InferRequestedOutput("search_terms__nnzs"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    print(
        "user_id: {}, movie_id_count: {}, movie_ids: {}, genres: {}, search_terms: {}".format(
            user_id_data,
            response.as_numpy("movie_id_count"),
            response.as_numpy("movie_ids__values"),
            response.as_numpy("genres__values"),
            response.as_numpy("search_terms__values"),
        )
    )
