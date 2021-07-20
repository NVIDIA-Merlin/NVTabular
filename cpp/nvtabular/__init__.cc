// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace nvtabular {
namespace inference { void export_module(py::module_); }

PYBIND11_MODULE(nvtabular_cpp, m) {
  m.doc() = R"nvtdoc(
    nvtabular_cpp
    -------------
    Provides C++ extensions for speeding up nvtabular workflows at inference time.
    )nvtdoc";

  py::module_ inference = m.def_submodule("inference");
  nvtabular::inference::export_module(inference);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace nvtabular
