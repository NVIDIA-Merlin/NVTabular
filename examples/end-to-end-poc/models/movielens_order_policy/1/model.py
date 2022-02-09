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
                # Extract parameters from the request
                candidate_ids = pb_utils.get_input_tensor_by_name(
                    request, "candidate_movie_ids").as_numpy().reshape(-1)

                predicted_scores = pb_utils.get_input_tensor_by_name(
                    request, "predicted_scores").as_numpy().reshape(-1)

                # Exponential sort trick for sampling from a distribution without replacement from:

                # Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir,
                # Information Processing Letters, Volume 97, Issue 5, 2006, Pages 181-185, ISSN 0020-0190,
                # https://doi.org/10.1016/j.ipl.2005.11.003.

                # As implented by Tim Vieira in "Algorithms for sampling without replacement"
                # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

                # TODO: Extract k and theta as config parameters
                k = 10
                theta = 20.0

                # The weights for the sampling distribution are the softmax of the scores
                weights = np.exp(theta * predicted_scores) / np.sum(predicted_scores)

                # This is the core of the exponential sampling trick, which creates a
                # set of values that depend on both the predicted scores and random
                # variables, resulting in a set of values that will sort into an order
                # that reflects sampling without replacement according to the weight
                # distribution
                num_items = candidate_ids.shape[0]
                exponentials = -np.log(np.random.uniform(0, 1, size=(num_items,)))
                exponentials /= weights

                # This is just bookkeeping to produce the final ordered list of recs
                sorted_indices = np.argsort(exponentials)
                topk_movie_ids = candidate_ids[sorted_indices][:k]
                ordered_movie_ids = topk_movie_ids.reshape(1, -1).astype(int32_dtype).T

                # And return it to the client
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("ordered_ids", ordered_movie_ids)
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
