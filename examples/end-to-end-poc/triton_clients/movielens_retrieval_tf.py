from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    user_id_data = np.random.randint(1, 1000, (1, 1)).astype(np.int32)
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

    response = client.infer("movielens_user_features", inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    movie_id_count = response.as_numpy("movie_id_count")
    movie_ids_values = response.as_numpy("movie_ids__values")
    movie_ids_nnzs = response.as_numpy("movie_ids__nnzs")
    genres_values = response.as_numpy("genres__values")
    genres_nnzs = response.as_numpy("genres__nnzs")
    search_terms_values = response.as_numpy("search_terms__values")
    search_terms_nnzs = response.as_numpy("search_terms__nnzs")

    movie_ids_nnzs[0][0] = movie_ids_values.shape[0]
    genres_nnzs[0][0] = genres_values.shape[0]
    search_terms_nnzs[0][0] = search_terms_values.shape[0]

    tf_inputs = [
        httpclient.InferInput(
            "movie_id_count", movie_id_count.shape, np_to_triton_dtype(movie_id_count.dtype)
        ),
        httpclient.InferInput(
            "movie_ids_1", movie_ids_values.shape, np_to_triton_dtype(movie_ids_values.dtype)
        ),
        httpclient.InferInput(
            "movie_ids_2", movie_ids_nnzs.shape, np_to_triton_dtype(movie_ids_nnzs.dtype)
        ),
        httpclient.InferInput(
            "genres_1", genres_values.shape, np_to_triton_dtype(genres_values.dtype)
        ),
        httpclient.InferInput("genres_2", genres_nnzs.shape, np_to_triton_dtype(genres_nnzs.dtype)),
        httpclient.InferInput(
            "search_terms_1",
            search_terms_values.shape,
            np_to_triton_dtype(search_terms_values.dtype),
        ),
        httpclient.InferInput(
            "search_terms_2", search_terms_nnzs.shape, np_to_triton_dtype(search_terms_nnzs.dtype)
        ),
    ]

    tf_inputs[0].set_data_from_numpy(movie_id_count)
    tf_inputs[1].set_data_from_numpy(movie_ids_values)
    tf_inputs[2].set_data_from_numpy(movie_ids_nnzs)
    tf_inputs[3].set_data_from_numpy(genres_values)
    tf_inputs[4].set_data_from_numpy(genres_nnzs)
    tf_inputs[5].set_data_from_numpy(search_terms_values)
    tf_inputs[6].set_data_from_numpy(search_terms_nnzs)

    tf_outputs = [
        httpclient.InferRequestedOutput("output_1"),
    ]

    tf_response = client.infer(
        "movielens_retrieval_tf", tf_inputs, request_id=str(2), outputs=tf_outputs
    )

    print(
        "user_id: {}, user_vector: {}".format(
            user_id_data,
            tf_response.as_numpy("output_1"),
        )
    )
