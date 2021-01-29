/*
 * nvtabular.h
 *
 *  Created on: Jan 13, 2021
 *      Author: oyilmaz
 */

#ifndef NVTABULAR_H_
#define NVTABULAR_H_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
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

	const void* get_data(py::object data, TRITONSERVER_DataType dtype) {
		if (dtype == TRITONSERVER_TYPE_BYTES) {

		} else if (dtype == TRITONSERVER_TYPE_INT8) {

		} else if (dtype == TRITONSERVER_TYPE_INT16) {

		} else if (dtype == TRITONSERVER_TYPE_INT32) {
			py::array_t<uint32_t> d = (py::array_t<uint32_t>) data;
			return d.data();
		} else if (dtype == TRITONSERVER_TYPE_INT64) {
			py::array_t<uint64_t> d = (py::array_t<uint64_t>) data;
			return d.data();
		} else if (dtype == TRITONSERVER_TYPE_FP16) {

		} else if (dtype == TRITONSERVER_TYPE_FP32) {
			py::array_t<float> d = (py::array_t<float>) data;
			return d.data();
		} else if (dtype == TRITONSERVER_TYPE_FP64) {
			py::array_t<double> d = (py::array_t<double>) data;
			return d.data();
		}
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

	void Transform(const char** input_names, const void** input_buffers,
			const int64_t** input_shapes, TRITONSERVER_DataType* input_dtypes,
			uint32_t input_count, void** output_buffers,
			uint64_t * output_byte_sizes, const char** output_names,
			uint32_t output_count,
			std::unordered_map<std::string, TRITONSERVER_DataType> &names_to_dtypes) {

		py::list all_inputs;
		py::list all_inputs_names;
		for (uint32_t i = 0; i < input_count; ++i) {
			py::dict ai_in;
			std::tuple<long> shape_in((long) input_shapes[i][0]);
			ai_in["shape"] = shape_in;
			std::tuple<long, bool> data_in((long) *(&input_buffers[i]), false);
			ai_in["data"] = data_in;
			fill_array_interface(ai_in, input_dtypes[i]);
			all_inputs.append(ai_in);
			all_inputs_names.append(input_names[i]);
		}

		py::dict output = nt.attr("transform")(all_inputs_names, all_inputs);

		for (uint32_t i = 0; i < output_count; ++i) {
			std::string curr_name(output_names[i]);
			TRITONSERVER_DataType dtype = names_to_dtypes[curr_name];
			if (dtype == TRITONSERVER_TYPE_INT32) {
				py::array_t<uint32_t> a = (py::array_t<uint32_t>) output[output_names[i]];
				memcpy (output_buffers[i], a.data(), output_byte_sizes[i]);

				std::cout << "** Col name: " << output_names[i] << ", Value: "
						<< a.data()[0] << std::endl;
			} else if (dtype == TRITONSERVER_TYPE_INT64) {
				py::array_t<uint64_t> a =
						(py::array_t<uint64_t>) output[output_names[i]];
				memcpy (output_buffers[i], a.data(), output_byte_sizes[i]);
				std::cout << "** Col name: " << output_names[i] << ", Value: "
						<< a.data()[0] << std::endl;
			} else if (dtype == TRITONSERVER_TYPE_FP16) {

			} else if (dtype == TRITONSERVER_TYPE_FP32) {
				py::array_t<float> a =
						(py::array_t<float>) output[output_names[i]];
				memcpy (output_buffers[i], a.data(), output_byte_sizes[i]);
				std::cout << "** Col name: " << output_names[i] << ", Value: "
						<< a.data()[0] << std::endl;
			} else {
				std::cout << "** None of them: " << output_names[i] << std::endl;
			}
		}

	}

private:
	py::object nt;
};

#endif /* NVTABULAR_H_ */
