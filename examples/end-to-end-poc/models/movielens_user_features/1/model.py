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

        self.entity_id = "user_id"
        self.entity_view = "user_features"
        self.features = ["movie_id_count"]
        self.mh_features = ["movie_ids", "genres", "search_terms"]

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
                entity_ids = pb_utils.get_input_tensor_by_name(request, self.entity_id).as_numpy()
                entity_rows = [{self.entity_id: int(entity_id)} for entity_id in entity_ids]

                feature_names = self.features + self.mh_features
                feature_refs = [
                    ":".join([self.entity_view, feature_name]) for feature_name in feature_names
                ]

                features = self.store.get_online_features(
                    feature_refs=feature_refs,
                    entity_rows=entity_rows,
                ).to_dict()

                output_tensors = []

                # Numerical and single-hot categorical
                for feature_name in self.features:
                    feature_value = features["__".join([self.entity_view, feature_name])]
                    feature_array = np.array([feature_value]).astype(int32_dtype).T

                    output_tensors.append(pb_utils.Tensor(feature_name, feature_array))

                # Multi-hot categorical
                for feature_name in self.mh_features:
                    feature_value = features["__".join([self.entity_view, feature_name])]
                    feature_array = np.array(feature_value).astype(int32_dtype).T

                    output_tensors.append(
                        pb_utils.Tensor("__".join([feature_name, "values"]), feature_array)
                    )
                    output_tensors.append(
                        pb_utils.Tensor(
                            "__".join([feature_name, "nnzs"]),
                            np.array([[len(feature_array)]], dtype=np.int32),
                        )
                    )

                responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
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
