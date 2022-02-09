import json
import sys
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils


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
        pass

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
                candidate_ids = pb_utils.get_input_tensor_by_name(
                    request, "candidate_movie_ids").as_numpy()

                # Parse the user feature inputs
                movie_id_count = pb_utils.get_input_tensor_by_name(
                    request, "movie_id_count").as_numpy()

                movie_ids_values = pb_utils.get_input_tensor_by_name(
                    request, "movie_ids__values").as_numpy()

                movie_ids_nnzs = pb_utils.get_input_tensor_by_name(
                    request, "movie_ids__nnzs").as_numpy()

                genres_values = pb_utils.get_input_tensor_by_name(
                    request, "genres__values").as_numpy()

                genres_nnzs = pb_utils.get_input_tensor_by_name(
                    request, "genres__nnzs").as_numpy()

                search_terms_values = pb_utils.get_input_tensor_by_name(
                    request, "search_terms__values").as_numpy()

                search_terms_nnzs = pb_utils.get_input_tensor_by_name(
                    request, "search_terms__nnzs").as_numpy()

                # Repeat the user features to match the size of the candidate ids
                num_items = candidate_ids.shape[0]

                movie_id_count_ur = np.repeat(movie_id_count, num_items, axis=0)
                movie_ids_values_ur = np.repeat(movie_ids_values, num_items, axis=0)
                movie_ids_nnzs_ur = np.repeat(movie_ids_nnzs, num_items, axis=0)
                genres_values_ur = np.repeat(genres_values, num_items, axis=0)
                genres_nnzs_ur = np.repeat(genres_nnzs, num_items, axis=0)
                search_terms_values_ur = np.repeat(search_terms_values, num_items, axis=0)
                search_terms_nnzs_ur = np.repeat(search_terms_nnzs, num_items, axis=0)

                # Return the repeated versions of the user features
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("movie_id_count_ur", movie_id_count_ur),
                            pb_utils.Tensor("movie_ids_values_ur", movie_ids_values_ur),
                            pb_utils.Tensor("movie_ids_nnzs_ur", movie_ids_nnzs_ur),
                            pb_utils.Tensor("genres_values_ur", genres_values_ur),
                            pb_utils.Tensor("genres_nnzs_ur", genres_nnzs_ur),
                            pb_utils.Tensor("search_terms_values_ur", search_terms_values_ur),
                            pb_utils.Tensor("search_terms_nnzs_ur", search_terms_nnzs_ur),
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
        pass
