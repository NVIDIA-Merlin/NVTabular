// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"
#include "nvtabular.h"
#include <dlfcn.h>
#include <unordered_map>
#include <chrono>
#include <rapidjson/document.h>
#include <iostream>
#include <fstream>
#include "model_state.h"
#include "model_inst_state.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

using namespace rapidjson;

namespace triton { namespace backend { namespace nvtabular {

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend. A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
//
// This backend supports any model that has exactly 1 input and
// exactly 1 output. The input and output can have any name, datatype
// and shape but the shape and datatype of the input and output must
// match. The backend simply responds with the output tensor equal to
// the input tensor.
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)


extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));


  RETURN_IF_ERROR(model_state->ReadInputOutputNames());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
	instance, reinterpret_cast<void**>(&instance_state)));

  ModelState* model_state = instance_state->StateForModel();

  std::string info = (std::string("model ") + model_state->Name() + ", instance " +
		    instance_state->Name() + ", executing " + std::to_string(request_count) +
		    " requests").c_str();
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, info.c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;
  std::string error = "";

  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
      responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
      responses, r,
      TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
      responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
      responses, r,
      TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    if (responses[r] == nullptr) {
      error = (std::string("request ") + std::to_string(r) +
              ": failed to read request input/output counts, error response sent")
                 .c_str();
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
      continue;
    }

    info = (std::string("request ") + std::to_string(r) + ": id = \"" +
            request_id + "\", correlation_id = " + std::to_string(correlation_id) +
            ", input_count = " + std::to_string(input_count) +
            ", requested_output_count = " + std::to_string(requested_output_count))
               .c_str();
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, info.c_str());

    const std::vector<std::string> input_names = model_state->InputNames();
    TRITONBACKEND_Input* inputs[input_count];
    TRITONSERVER_DataType input_dtypes[input_count];
    const int64_t* input_shapes[input_count];
    uint32_t input_dims_counts[input_count];
    uint64_t input_byte_sizes[input_count];
    uint32_t input_buffer_counts[input_count];

    for (uint32_t i = 0; i < input_count; i++) {
      const char* input_name = input_names[i].c_str();
      GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name, &inputs[i]));

      GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
          inputs[i], &input_name, &input_dtypes[i], &input_shapes[i],
          &input_dims_counts[i], &input_byte_sizes[i], &input_buffer_counts[i]));

      if (responses[r] == nullptr) {
    	error = (std::string("request ") + std::to_string(r) +
                ": failed to read input properties, error response sent")
                   .c_str();
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
        continue;
      }

      /*
      info = (std::string("\tinput ") + input_name +
              ": datatype = " + TRITONSERVER_DataTypeString(input_dtypes[i]) +
              ", shape = " + backend::ShapeToString(input_shapes[i], input_dims_counts[i]) +
              ", byte_size = " + std::to_string(input_byte_sizes[i]) +
              ", buffer_count = " + std::to_string(input_buffer_counts[i]))
                 .c_str();
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, info.c_str());
      */
    }

    const std::vector<std::string> output_names = model_state->OutputNames();
    const std::vector<TRITONSERVER_DataType> output_dtypes = model_state->OutputDtypes();
    TRITONBACKEND_Output* outputs[output_names.size()];
    void* output_buffers[output_names.size()];
    uint64_t output_byte_sizes[output_names.size()];
    std::vector<std::vector<wchar_t>*> numpy_input_buffers;
    std::unordered_map<std::string, size_t> max_str_sizes;

    for (uint32_t b = 0; b < input_buffer_counts[0]; ++b) {
      const void* input_buffers[input_count];
      uint64_t buffer_byte_sizes[input_count];

      for (uint32_t i = 0; i < input_count; ++i) {
    	input_buffers[i] = nullptr;

    	buffer_byte_sizes[i] = 0;
    	TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    	int64_t input_memory_type_id = 0;

    	const void* input_buffer;

    	GUARDED_RESPOND_IF_ERROR(
    	  responses, r,
    	  TRITONBACKEND_InputBuffer(
    	    inputs[i], b, &input_buffer, &buffer_byte_sizes[i], &input_memory_type,
    	    &input_memory_type_id));

    	if (input_dtypes[i] == TRITONSERVER_TYPE_BYTES) {
    	  size_t max_size = Utils::GetMaxStringLen((const unsigned char*)input_buffer, buffer_byte_sizes[i]);

    	  max_str_sizes[input_names[i]] = max_size;
    	  size_t nif_size = max_size * input_shapes[i][0];
    	  std::vector<wchar_t>* numpy_input_buffer = new std::vector<wchar_t>(nif_size, '\0');
    	  numpy_input_buffers.push_back(numpy_input_buffer);

    	  Utils::ConstructNumpyStringArray(numpy_input_buffer->data(), (uint64_t)max_size,
    			  (const unsigned char*)input_buffer, buffer_byte_sizes[i]);
    	  input_buffers[i] = numpy_input_buffer->data();
    	} else {
    	  input_buffers[i] = input_buffer;
    	}

    	if ((responses[r] == nullptr) ||
    	  (input_memory_type == TRITONSERVER_MEMORY_GPU)) {

    		GUARDED_RESPOND_IF_ERROR(
    	      responses, r,
    	      TRITONSERVER_ErrorNew(
    	        TRITONSERVER_ERROR_UNSUPPORTED,
    	        "failed to get input buffer in CPU memory"));
    	}
      }

      instance_state->nvt.Transform(input_names, input_buffers, input_shapes,
        input_dtypes, max_str_sizes, output_names);

      py::list lengths = instance_state->nvt.GetOutputSizes();
      for (uint32_t i = 0; i < output_names.size(); ++i) {
    	const char* output_name = output_names[i].c_str();
    	int64_t output_length = lengths[i].cast<int64_t>();
    	int64_t output_width = 1;
    	output_byte_sizes[i] = output_length * output_width *
    			Utils::GetTritonTypeByteSize(output_dtypes[i]);

    	std::vector<int64_t> batch_shape;
    	batch_shape.push_back(output_length);
    	batch_shape.push_back(output_width);

    	TRITONBACKEND_Response* response = responses[r];
    	  GUARDED_RESPOND_IF_ERROR(
    	    responses, r,
    	    TRITONBACKEND_ResponseOutput(
    	      response, &outputs[i], output_name, output_dtypes[i],
    	      batch_shape.data(), batch_shape.size()));

    	if (responses[r] == nullptr) {
          error = (std::string("request ") + std::to_string(r) +
                  ": failed to create response output, error response sent").c_str();
    	  LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
    	  continue;
    	}

    	TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t output_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
          outputs[i], &output_buffers[i], output_byte_sizes[i], &output_memory_type,
          &output_memory_type_id));

        if ((responses[r] == nullptr) || (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
          GUARDED_RESPOND_IF_ERROR(
             responses, r,
             TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
             "failed to create output buffer in CPU memory"));

          info = (std::string("request ") + std::to_string(r) +
                 ": failed to create output buffer in CPU memory, error response "
                 "sent").c_str();
    	  LOG_MESSAGE(TRITONSERVER_LOG_ERROR, info.c_str());
    	          continue;
        }

      }

      instance_state->nvt.CopyData(output_buffers, output_byte_sizes, output_names, output_dtypes);

      if (responses[r] == nullptr) {
        error = (std::string("request ") + std::to_string(r) +
       	        ": failed to get input buffer in CPU memory, error response "
       	        "sent").c_str();
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
        continue;
      }
    }

    if (supports_batching && (input_dims_counts[0] > 0)) {
      total_batch_size += input_shapes[0][0];
    } else {
      total_batch_size++;
    }

    for (size_t i = 0; i < numpy_input_buffers.size(); ++i) {
      delete numpy_input_buffers[i];
    }

    LOG_IF_ERROR(
      TRITONBACKEND_ResponseSetStringParameter(
        responses[r], "param0", "an example string parameter"),
        "failed setting string parameter");

    LOG_IF_ERROR(
      TRITONBACKEND_ResponseSetIntParameter(responses[r], "param1", 42),
        "failed setting integer parameter");

    LOG_IF_ERROR(
      TRITONBACKEND_ResponseSetBoolParameter(responses[r], "param2", false),
        "failed setting boolean parameter");

    LOG_IF_ERROR(
      TRITONBACKEND_ResponseSend(
        responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        nullptr /* success */),
        "failed sending response");

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

    LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportStatistics(
        instance_state->TritonModelInstance(), request, true /* success */,
        exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  LOG_IF_ERROR(
    TRITONBACKEND_ModelInstanceReportBatchStatistics(
      instance_state->TritonModelInstance(), total_batch_size,
      min_exec_start_ns, min_exec_start_ns, max_exec_end_ns,
      max_exec_end_ns),
      "failed reporting batch request statistics");

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    if (responses[r] == nullptr) {
      LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
           instance_state->TritonModelInstance(), request,
           false /* success */, 0, 0, 0, 0),
           "failed reporting request statistics");
    }

    LOG_IF_ERROR(
      TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  //LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Request successfully completed");
  return nullptr;  // success
}

}  // extern "C"

}}}
