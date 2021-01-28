/*
 * nvtabular.h
 *
 *  Created on: Jan 13, 2021
 *      Author: oyilmaz
 */

#ifndef NVTABULAR_H_
#define NVTABULAR_H_

#include <pybind11/embed.h>
#include "triton/backend/backend_common.h"

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
		py::object nvtabular = py::module_::import("nvt").attr("nvt");
		nt = nvtabular();
		nt.attr("deserialize")(path.data());
	}

	void Transform(const void** input_buffers, const int64_t** input_shapes,
			uint64_t* buffer_byte_sizes, TRITONSERVER_DataType* input_dtypes,
			uint32_t input_count, void** output_buffers,
			uint32_t output_count) {

		py::list all_inputs;
		for (uint32_t i = 0; i < input_count; ++i) {
			py::dict ai_in;
			std::tuple<long> shape_in((long) input_shapes[i][0]);
			ai_in["shape"] = shape_in;
			std::tuple<long, bool> data_in((long) *(&input_buffers[i]), false);
			ai_in["data"] = data_in;
			fill_array_interface(ai_in, input_dtypes[i]);
			all_inputs.append(ai_in);
		}

		py::list all_outputs;
		for (uint32_t i = 0; i < output_count; ++i) {
			py::dict ai_out;
			std::tuple<long> shape_out((long) input_shapes[i][0]);
			ai_out["shape"] = shape_out;
			std::tuple<long, bool> data_out((long) *(&output_buffers[i]),
					false);
			ai_out["data"] = data_out;
			all_outputs.append(ai_out);
		}

		nt.attr("transform")(all_inputs, all_outputs);

	}

private:
	py::object nt;
};

#endif /* NVTABULAR_H_ */
