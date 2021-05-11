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

#ifndef MODELINSTSTATE_H_
#define MODELINSTSTATE_H_

#include "utils.h"
#include <map>
#include <vector>

using namespace rapidjson;

namespace triton {
namespace backend {
namespace nvtabular {

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
public:
  static TRITONSERVER_Error *
  Create(ModelState *model_state,
         TRITONBACKEND_ModelInstance *triton_model_instance,
         ModelInstanceState **state);

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance *TritonModelInstance() {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string &Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState *StateForModel() const { return model_state_; }

  bool inter_started = false;
  NVTabular nvt;

private:
  ModelInstanceState(ModelState *model_state,
                     TRITONBACKEND_ModelInstance *triton_model_instance,
                     const char *name,
                     const TRITONSERVER_InstanceGroupKind kind,
                     const int32_t device_id);

  ModelState *model_state_;
  TRITONBACKEND_ModelInstance *triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
};

TRITONSERVER_Error *
ModelInstanceState::Create(ModelState *model_state,
                           TRITONBACKEND_ModelInstance *triton_model_instance,
                           ModelInstanceState **state) {
  const char *instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(model_state, triton_model_instance,
                                  instance_name, instance_kind, instance_id);

  std::string path_workflow(model_state->Path());
  path_workflow.append("/");
  path_workflow.append(std::to_string(model_state->Version()));
  path_workflow.append("/workflow");

  const std::vector<TRITONSERVER_DataType> triton_dtypes =
      model_state->OutputDtypes();
  std::map<std::string, std::string> dtypes;
  for (size_t i = 0; i < model_state->OutputNames().size(); i++) {
    dtypes[model_state->OutputNames()[i]] =
        Utils::ConvertToNumpyType(triton_dtypes[i]);
  }

  (*state)->nvt.Deserialize(path_workflow, dtypes);
  return nullptr; // success
}

ModelInstanceState::ModelInstanceState(
    ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
    const char *name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id) {}

} // namespace nvtabular
} // namespace backend
} // namespace triton

#endif /* MODELINSTSTATE_H_ */
