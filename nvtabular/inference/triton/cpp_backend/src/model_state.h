// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MODELSTATE_H_
#define MODELSTATE_H_

#include "triton/backend/backend_common.h"
#include "utils.h"
#include <vector>

using namespace rapidjson;

namespace triton {
namespace backend {
namespace nvtabular {
//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model *TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string &Name() const { return name_; }
  uint64_t Version() const { return version_; }
  const std::string &Path() const { return path_; }
  const std::vector<std::string> &InputNames() { return input_names_; }
  const std::vector<std::string> &OutputNames() { return output_names_; }
  const std::vector<TRITONSERVER_DataType> &OutputDtypes() {
    return output_dtypes_;
  }

  // Does this model support batching in the first dimension. This
  // function should not be called until after the model is completely
  // loaded.
  TRITONSERVER_Error *SupportsFirstDimBatching(bool *supports);

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error *CreationDelay();

  TRITONSERVER_Error *ReadInputOutputNames();

private:
  ModelState(TRITONSERVER_Server *triton_server,
             TRITONBACKEND_Model *triton_model, const char *name,
             const uint64_t version, const char *path,
             common::TritonJson::Value &&model_config);

  TRITONSERVER_Server *triton_server_;
  TRITONBACKEND_Model *triton_model_;
  const std::string name_;
  const uint64_t version_;
  const std::string path_;
  common::TritonJson::Value model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<TRITONSERVER_DataType> output_dtypes_;
};

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model,
                                       ModelState **state) {
  TRITONSERVER_Message *config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char *buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error *err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char *model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server *triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  TRITONBACKEND_ArtifactType artifact_type;
  const char *path;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));

  std::string python_version_path(path);
  python_version_path.append("/");
  python_version_path.append(std::to_string(model_version));
  python_version_path.append("/workflow/metadata.json");

  std::string line;
  std::ifstream myfile(python_version_path.c_str());
  if (myfile.is_open()) {
    std::getline(myfile, line);
    myfile.close();
  }

  Document document;
  document.Parse(line.c_str());
  if (document["versions"].HasMember("python")) {
    std::string python_lib = "libpython";

    std::string value(document["versions"]["python"].GetString());
    python_lib.append(value.substr(0, 3));
    python_lib.append(".so");

    void *handle = dlopen(python_lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, dlerror());
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Loaded libpython successfully");
    }
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                "Python version is not specified in the metada.json");
  }

  *state = new ModelState(triton_server, triton_model, model_name,
                          model_version, path, std::move(model_config));
  return nullptr; // success
}

ModelState::ModelState(TRITONSERVER_Server *triton_server,
                       TRITONBACKEND_Model *triton_model, const char *name,
                       const uint64_t version, const char *path,
                       common::TritonJson::Value &&model_config)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), path_(path), model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false) {}

TRITONSERVER_Error *ModelState::SupportsFirstDimBatching(bool *supports) {
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr; // success
}

TRITONSERVER_Error *ModelState::CreationDelay() {
  // Feature for testing purpose...
  // look for parameter 'creation_delay_sec' in model config
  // and sleep for the value specified
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value creation_delay_sec;
    if (parameters.Find("creation_delay_sec", &creation_delay_sec)) {
      std::string creation_delay_sec_str;
      RETURN_IF_ERROR(creation_delay_sec.MemberAsString(
          "string_value", &creation_delay_sec_str));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Creation delay is set to : ") + creation_delay_sec_str)
              .c_str());
      std::this_thread::sleep_for(
          std::chrono::seconds(std::stoi(creation_delay_sec_str)));
    }
  }
  return nullptr; // success
}

TRITONSERVER_Error *ModelState::ReadInputOutputNames() {
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  for (size_t i = 0; i < inputs.ArraySize(); i++) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));

    std::string input_name;
    input.MemberAsString("name", &input_name);
    input_names_.push_back(input_name);
  }

  for (size_t i = 0; i < outputs.ArraySize(); i++) {
    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    std::string output_name;
    output.MemberAsString("name", &output_name);
    std::string output_dtype;
    output.MemberAsString("data_type", &output_dtype);

    output_dtypes_.push_back(Utils::ConvertToTritonType(output_dtype));
    output_names_.push_back(output_name);
  }

  return nullptr; // success
}

} // namespace nvtabular
} // namespace backend
} // namespace triton

#endif /* MODELSTATE_H_ */
