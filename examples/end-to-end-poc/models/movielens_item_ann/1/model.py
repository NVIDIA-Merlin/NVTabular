import json
import sys
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils

import pymilvus_orm
from pymilvus_orm import schema, DataType, Collection


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
        self.index_name = self.model_config["parameters"]["index_name"]["string_value"]

        self.milvus_client = pymilvus_orm.connections.connect(
            host="milvus-standalone", port="19530")

        dim = 128
        default_fields = [
            schema.FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True),
            schema.FieldSchema(name="item_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        default_schema = schema.CollectionSchema(
            fields=default_fields, description="MovieLens item vectors")
        self.milvus_collection = Collection(
            name="movielens_retrieval_tf", data=None, schema=default_schema)

        self.milvus_collection.load()

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
                user_vector = pb_utils.get_input_tensor_by_name(
                    request, "user_vector").as_numpy()

                # Normalize the user query vector
                user_vector = user_vector / np.sqrt(np.sum(user_vector**2))

                topK = 100
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                result = self.milvus_collection.search(
                    user_vector, "item_vector", search_params, topK)

                candidate_ids = (
                    np.array([[h.id for hits in result for h in hits]]).astype(int32_dtype).T
                )
                candidate_distances = np.array(
                    [[h.distance for hits in result for h in hits]]).astype(int32_dtype).T

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("candidate_ids", candidate_ids),
                            pb_utils.Tensor("candidate_distances", candidate_distances),
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
        # self.milvus_client.disconnect()
