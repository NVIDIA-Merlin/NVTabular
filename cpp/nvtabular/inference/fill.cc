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
#include <pybind11/numpy.h>

#include <mutex>
#include <memory>
#include <map>
#include <unordered_map>
#include <iostream>

namespace nvtabular {
namespace inference {

namespace py = pybind11;

// Accelerated fill operator for running under triton inference server
struct FillTransform {
  explicit FillTransform(py::object op) {
    add_binary_cols = py::cast<bool>(op.attr("add_binary_cols"));
    fill_val = py::cast<double>(op.attr("fill_val"));
  }

  py::object transform(py::list column_names, py::dict tensors) {
    py::dict ret;
    for (auto it: tensors) {
      auto column_name = py::cast<std::string>(it.first);
      auto tensor = py::cast<py::array>(it.second);

      auto dtype = tensor.dtype();

      if (!tensor.writeable()) {
        tensor = tensor.attr("copy")();
      }

      switch (dtype.kind()) {
        case 'f': {
          switch (dtype.itemsize()) {
            case 4: {
              py::array_t<float> values(tensor);
              fill<float>(values.mutable_data(), values.size());
              ret[column_name.c_str()] = values;
              continue;
            }
            case 8: {
              py::array_t<double> values(tensor);
              fill<double>(values.mutable_data(), values.size());
              ret[column_name.c_str()] = values;
              continue;
            }
            default: {
              std::stringstream err;
              err << "Unhandled floating point width for FillTransform" << dtype.itemsize();
              throw std::invalid_argument(err.str());
            }
          }
        }
      }
      // we can't really fill non-floating point columns (integers don't have 'nan' really)
      // so just copy over
      ret[column_name.c_str()] = tensor;
    }
    return ret;
  }

  template <typename T>
  void fill(T * values, size_t size) {
    T fill = static_cast<T>(fill_val);
    for (size_t i = 0; i < size; ++i) {
      if (isnan(values[i])) {
        values[i] = fill;
      }
    }
  }

  double fill_val;
  bool add_binary_cols;
};

void export_fill(py::module_ m) {
  py::class_<FillTransform>(m, "FillTransform")
    .def(py::init<py::object>())
    // this operator currently only supports CPU arrays
    .def_property_readonly("supports", [](py::object self) {
      py::object supports = py::module_::import("nvtabular").attr("ops").attr("operator").attr("Supports");
      return supports.attr("CPU_DICT_ARRAY");
    })
    .def("transform", &FillTransform::transform);
}
}  // namespace inference
}  // namespace nvtabular
