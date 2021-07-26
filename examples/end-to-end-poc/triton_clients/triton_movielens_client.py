import argparse

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "movielens"
shape = (1, 1)

parser = argparse.ArgumentParser(description='Fetch recommendations for a user')
parser.add_argument('--user_id', type=int, help='ID of the user')
args = parser.parse_args()


with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    # user_id_data = np.random.randint(1, 100000, shape).astype(np.int32)
    user_id_data = np.array([[args.user_id]]).astype(np.int32)

    inputs = [
        httpclient.InferInput(
            "user_id", user_id_data.shape, np_to_triton_dtype(user_id_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(user_id_data)

    outputs = [
        httpclient.InferRequestedOutput("filtered_candidate_ids"),
        httpclient.InferRequestedOutput("predicted_scores"),
        httpclient.InferRequestedOutput("ordered_movie_ids")
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()

    filtered_candidate_ids = response.as_numpy("filtered_candidate_ids").reshape(-1)
    predicted_scores = response.as_numpy("predicted_scores").reshape(-1)
    ordered_movie_ids = response.as_numpy("ordered_movie_ids").reshape(-1)

    sorted_indices = np.flip(np.argsort(predicted_scores))

    print(
        "\ncandidate ids:\n{}\n\ncandidate scores:\n{}\n\nrecommended items:\n{}\n".format(
            filtered_candidate_ids[sorted_indices],
            predicted_scores[sorted_indices],
            ordered_movie_ids,
        )
    )
