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

#ifndef NVTABULAR_H_
#define NVTABULAR_H_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include "triton/backend/backend_common.h"
#include <exception>
#include <vector>

namespace py = pybind11;

class NVTabular {

private:
	void fill_array_interface(py::dict &ai, TRITONSERVER_DataType dtype) {
		py::list list_desc;
		if (dtype == TRITONSERVER_TYPE_BYTES) {
			ai["typestr"] = "<U6";
			std::tuple<std::string, std::string> desc("", "<U6");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_INT8) {
			ai["typestr"] = "<i1";
			std::tuple<std::string, std::string> desc("", "<i1");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_INT16) {
			ai["typestr"] = "<i2";
			std::tuple<std::string, std::string> desc("", "<i2");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_INT32) {
			ai["typestr"] = "<i4";
			std::tuple<std::string, std::string> desc("", "<i4");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_INT64) {
			ai["typestr"] = "<i8";
			std::tuple<std::string, std::string> desc("", "<i8");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_FP16) {
			ai["typestr"] = "<f2";
			std::tuple<std::string, std::string> desc("", "<f2");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_FP32) {
			ai["typestr"] = "<f4";
			std::tuple<std::string, std::string> desc("", "<f4");
			list_desc.append(desc);
		} else if (dtype == TRITONSERVER_TYPE_FP64) {
			ai["typestr"] = "<f8";
			std::tuple<std::string, std::string> desc("", "<f8");
			list_desc.append(desc);
		}
		ai["descr"] = list_desc;
		ai["version"] = 3;
	}
public:

	NVTabular() {
		py::initialize_interpreter();
		LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python interpreter is initialized");
	}

	~NVTabular() {
		py::finalize_interpreter();
		LOG_MESSAGE(TRITONSERVER_LOG_INFO,
				"Python interpreter is  finalized\n");
	}

	void Deserialize(std::string path) {
		py::object nvtabular = py::module_::import("nvtabular.inference.triton.cpp_backend.nvt").attr("TritonNVTabularModel");
		nt = nvtabular();
		nt.attr("initialize")(path.data());
	}

	void Transform(const std::vector<std::string>& input_names, const void** input_buffers,
			const int64_t** input_shapes, TRITONSERVER_DataType* input_dtypes,
			const std::vector<std::string>& output_names) {

		py::list all_inputs;
		py::list all_inputs_names;
		for (uint32_t i = 0; i < input_names.size(); ++i) {
			py::dict ai_in;
			std::tuple<long> shape_in((long) input_shapes[i][0]);
			ai_in["shape"] = shape_in;
			std::tuple<long, bool> data_in((long) *(&input_buffers[i]), false);
			ai_in["data"] = data_in;
			fill_array_interface(ai_in, input_dtypes[i]);
			all_inputs.append(ai_in);
			all_inputs_names.append(input_names[i]);
		}

		py::list all_output_names;
		for (uint32_t i = 0; i < output_names.size(); ++i) {
			all_output_names.append(output_names[i]);
		}

		auto transform_start = std::chrono::high_resolution_clock::now();

		py::dict output = nt.attr("transform")(all_inputs_names, all_inputs, all_output_names);

		auto transform_end = std::chrono::high_resolution_clock::now();
		auto elapsed_transform = std::chrono::duration_cast<
				std::chrono::nanoseconds>(transform_end - transform_start);
		printf("Transform Only Time measured: %.3f seconds.\n",
				elapsed_transform.count() * 1e-9);

		/*
		for (uint32_t i = 0; i < output_count; ++i) {
			std::string curr_name(output_names[i]);
			TRITONSERVER_DataType dtype = names_to_dtypes[curr_name];
			if (dtype == TRITONSERVER_TYPE_INT32) {
				py::array_t<uint32_t> a =
						(py::array_t<uint32_t>) output[output_names[i]];
				memcpy(output_buffers[i], a.data(), output_byte_sizes[i]);

			} else if (dtype == TRITONSERVER_TYPE_INT64) {
				py::array_t<uint64_t> a =
						(py::array_t<uint64_t>) output[output_names[i]];
				memcpy(output_buffers[i], a.data(), output_byte_sizes[i]);
			} else if (dtype == TRITONSERVER_TYPE_FP16) {

			} else if (dtype == TRITONSERVER_TYPE_FP32) {
				py::array_t<float> a =
						(py::array_t<float>) output[output_names[i]];
				memcpy(output_buffers[i], a.data(), output_byte_sizes[i]);
			} else {
				std::cout << "** None of them: " << output_names[i]
						<< std::endl;
			}
		}
		*/
	}

private:
	py::object nt;
};

#endif /* NVTABULAR_H_ */
