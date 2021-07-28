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

struct ColumnMapping {
  explicit ColumnMapping(const std::string & column_name, const std::string & filename)
      : filename(filename) {
    // use pandas to read into a dataframe. Note: we're purposefully doing this on the
    // CPU here to avoid using gpu memory at inference time
    py::object df = py::module_::import("pandas").attr("read_parquet")(filename);
    py::array values = df[column_name.c_str()].attr("values");
    auto dtype = values.dtype();

    if ((dtype.kind() == 'O') || (dtype.kind() == 'U')) {
      int64_t i = 0;
      for (auto & value: values) {
        if (!value.is_none()) {
          if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr())) {
            std::string key = py::cast<std::string>(value);
            mapping_str[key] = i;
          } else if (PyBool_Check(value.ptr())) {
            // so we're categorifying bool columns in the rossmann dataset - which makes little
            // sense but we have to handle I guess ?
            mapping_int[value.ptr() == Py_True] = i;
          } else {
            std::stringstream error;
            error << "Don't know how to handle column " << column_name << " @ " << filename;
            throw std::invalid_argument(error.str());
          }
        }
        i++;
      }
    } else {
      // TODO: array dispatch code
      switch (dtype.kind()) {
        case 'f':
          switch (dtype.itemsize()) {
            case 4:
              insert_int_mapping<float>(values);
              return;
            case 8:
              insert_int_mapping<double>(values);
              return;
          }
          break;
        case 'u':
          switch (dtype.itemsize()) {
            case 4:
              insert_int_mapping<uint32_t>(values);
              return;
            case 8:
              insert_int_mapping<uint64_t>(values);
              return;
          }
          break;
        case 'i':
          switch (dtype.itemsize()) {
            case 4:
              insert_int_mapping<int32_t>(values);
              return;
            case 8:
              insert_int_mapping<int64_t>(values);
              return;
          }
          break;
      }
      std::stringstream error;
      error << "unhandled dtype " << dtype.kind() << dtype.itemsize() << " for column " << column_name;
      throw std::invalid_argument(error.str());
    }
  }

  template <typename T>
  void insert_int_mapping(py::array_t<T> values) {
    const T * data = values.data();
    size_t size = values.size();
    for (size_t i = 0; i < size; ++i) {
      mapping_int[static_cast<int64_t>(data[i])] = i;
    }
  }

  template <typename T>
  py::array transform_int(py::array_t<T> input) const {
    py::array_t<int64_t> output(input.size());
    const T * input_data = input.data();
    int64_t * output_data = output.mutable_data();
    for (int64_t i = 0; i < input.size(); ++i) {
      auto it = mapping_int.find(static_cast<int64_t>(input_data[i]));
      output_data[i] = it == mapping_int.end() ? 0 : it->second;
    }
    return output;
  }

  py::array transform(py::array input) const {
    auto dtype = input.dtype();
    auto kind = dtype.kind();
    if ((kind == 'O') || (kind == 'U')) {
      size_t i = 0;
      py::array_t<int64_t> output(input.size());
      int64_t * data = output.mutable_data();
      for (auto & value: input) {
        if (value.is_none()) {
          data[i] = 0;
        } else if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr())) {
          std::string key = py::cast<std::string>(value);
          auto it = mapping_str.find(key);
          data[i] = it == mapping_str.end() ? 0 : it->second;
        } else if (PyBool_Check(value.ptr())) {
          auto it = mapping_int.find(value.ptr() == Py_True);
          data[i] = it == mapping_int.end() ? 0 : it->second;
        } else {
          throw std::invalid_argument("unknown dtype");
        }
        i++;
      }
      return output;
    } else {
      // TODO: array dispatch code
      auto itemsize = dtype.itemsize();
      switch (kind) {
        // floats don't really make sense here, but can happen because of auto
        // conversion with pandas code involving none values
        case 'f':
          switch (itemsize) {
            case 4: return transform_int<float>(input);
            case 8: return transform_int<double>(input);
          }
          break;
        case 'u':
          switch (itemsize) {
            case 4: return transform_int<uint32_t>(input);
            case 8: return transform_int<uint64_t>(input);
          }
          break;
        case 'i':
          switch (itemsize) {
            case 4: return transform_int<int32_t>(input);
            case 8: return transform_int<int64_t>(input);
          }
          break;
        case 'b':
          return transform_int<char>(input);
      }
      std::stringstream error;
      error << "unhandled dtype " << kind << itemsize << " for column '" << column_name << "'";
      throw std::invalid_argument(error.str());
    }
  }

  std::string filename;
  std::string column_name;

  std::unordered_map<std::string, int64_t> mapping_str;
  std::unordered_map<int64_t, int64_t> mapping_int;
};

// Reads in a parquet category mapping file in cpu memory using pandas
std::shared_ptr<ColumnMapping> get_column_mapping(const std::string & column_name,
                                                  const std::string & filename) {
  // because of how we're doing multi-gpu inside of tritonserver, we could have
  // multiple instances of the same workflow running in the same process. (with
  // each workflow having unique per-gpu data).
  // Since we're storing this mapping in CPU memory, lets cache values across
  // processes to reduce duplicate memory usage.
  static std::map<std::string, std::weak_ptr<ColumnMapping>> cache;
  static std::mutex m;

  std::lock_guard<std::mutex> lock(m);
  auto cache_item = cache[filename].lock();
  if (!cache_item) {
    cache[filename] = cache_item = std::make_shared<ColumnMapping>(column_name, filename);
  }
  return cache_item;
}


// Accelerated categorify operator for running under triton inference server
struct CategorifyTransform {
  explicit CategorifyTransform(py::object op) {
    py::dict categories = op.attr("categories");
    for (auto & item : categories) {
      auto column = py::cast<std::string>(item.first);
      auto filename = py::cast<std::string>(item.second);
      columns[column] = get_column_mapping(column, filename);
    }
  }

  py::object transform(py::list column_names, py::dict tensors) {
    for (auto it: tensors) {
      auto column_name = py::cast<std::string>(it.first);
      auto mapping = columns.find(column_name);
      if (mapping == columns.end()) {
        std::stringstream err;
        err << "Unknown column for CategorifyTransform "  << column_name;
        throw std::invalid_argument(err.str());
      }

      if (PyTuple_Check(it.second.ptr())) {
        auto value_offsets = py::cast<py::tuple>(it.second);
        auto tensor = py::cast<py::array>(value_offsets[0]);
        tensors[column_name.c_str()] = py::make_tuple(mapping->second->transform(tensor), value_offsets[1]);
      } else {
        auto tensor = py::cast<py::array>(it.second);
        tensors[column_name.c_str()] = mapping->second->transform(tensor);
      }
    }
    return tensors;
  }
  std::unordered_map<std::string, std::shared_ptr<ColumnMapping>> columns;
};


void export_categorify(py::module_ m) {
  py::class_<CategorifyTransform>(m, "CategorifyTransform")
    .def(py::init<py::object>())
    .def("transform", &CategorifyTransform::transform)

    // this operator currently only supports CPU arrays
    .def_property_readonly("supports", [](py::object self) {
      py::object supports = py::module_::import("nvtabular").attr("ops").attr("operator").attr("Supports");
      return supports.attr("CPU_DICT_ARRAY");
    });
}
}  // namespace inference
}  // namespace nvtabular
