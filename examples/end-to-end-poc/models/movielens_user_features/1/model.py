import json
import sys
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils

from feast import FeatureStore


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        repo_path = self.model_config["parameters"]["feast_repo_path"]["string_value"]

        self.store = FeatureStore(repo_path=repo_path)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        int32_dtype = pb_utils.triton_string_to_numpy("TYPE_INT32")
        # int64_dtype = pb_utils.triton_string_to_numpy("TYPE_INT64")

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            try:
                user_ids = pb_utils.get_input_tensor_by_name(request, "user_id").as_numpy()

                entity_rows = [{"user_id": int(user_id)} for user_id in user_ids]

                feature_vector = self.store.get_online_features(
                    feature_refs=[
                        "user_features:search_terms",
                        "user_features:genres",
                        "user_features:movie_ids",
                    ],
                    entity_rows=entity_rows,
                ).to_dict()

                movie_ids = (
                    np.array(feature_vector["user_features__movie_ids"]).astype(int32_dtype).T
                )
                genres = np.array(feature_vector["user_features__genres"]).astype(int32_dtype).T
                search_terms = (
                    np.array(feature_vector["user_features__search_terms"]).astype(int32_dtype).T
                )

                movie_id_count = pb_utils.Tensor(
                    "movie_id_count",
                    np.array([[len(movie_ids)]], dtype=np.int32),
                )

                movie_ids_values = pb_utils.Tensor(
                    "movie_ids__values",
                    movie_ids,
                )

                movie_ids_nnzs = pb_utils.Tensor(
                    "movie_ids__nnzs",
                    np.array([[len(movie_ids)]], dtype=np.int32),
                )

                genres_values = pb_utils.Tensor(
                    "genres__values",
                    genres,
                )

                genres_nnzs = pb_utils.Tensor(
                    "genres__nnzs",
                    np.array([[len(genres)]], dtype=np.int32),
                )

                search_terms_values = pb_utils.Tensor(
                    "search_terms__values",
                    search_terms,
                )

                search_terms_nnzs = pb_utils.Tensor(
                    "search_terms__nnzs",
                    np.array([[len(search_terms)]], dtype=np.int32),
                )

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            movie_id_count,
                            movie_ids_values,
                            movie_ids_nnzs,
                            genres_values,
                            genres_nnzs,
                            search_terms_values,
                            search_terms_nnzs,
                        ]
                    )
                )
            except Exception as e:
                exc = sys.exc_info()
                formatted_tb = str(traceback.format_tb(exc[-1]))
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"{exc[0]}, {exc[1]}, {formatted_tb}"),
                    )
                )

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
