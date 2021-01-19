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
public:

	NVTabular() {
		py::initialize_interpreter();
		LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python interpreter is initialized");
	}

	~NVTabular() {
		Py_DECREF(object);
		py::finalize_interpreter();
		LOG_MESSAGE(TRITONSERVER_LOG_INFO,
				"Python interpreter is  finalized\n");
	}

	void Deserialize(std::string path) {
		PyObject *module_name, *module, *dict, *python_class;
		module_name = PyUnicode_DecodeFSDefault("nvt");

		module = PyImport_Import(module_name);
		if (module == nullptr) {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
					"Fails to import the module nvt");
			PyErr_Print();
		}

		dict = PyModule_GetDict(module);
		if (dict == nullptr) {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
					"Fails to get the dictionary for module nvt");
			PyErr_Print();
		}

		python_class = PyDict_GetItemString(dict, "nvtabular");
		if (python_class == nullptr) {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
					"Fails to get the Python class nvtabular");
			PyErr_Print();
		}

		if (PyCallable_Check(python_class)) {
			object = PyObject_CallObject(python_class, nullptr);
		} else {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
					"Cannot instantiate the Python class");
			PyErr_Print();
		}

		PyObject *res = PyObject_CallMethod(object, "deserialize", "(i)", 2);
		if (!res) {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
					"Cannot instantiate the Python class");
			PyErr_Print();
		}

		Py_DECREF(module_name);
		Py_DECREF(module);
		Py_DECREF(dict);
		Py_DECREF(python_class);
	}

	void Transform(float* cont_in, int n_cont_in_rows, int n_cont_in_cols,
			float* cont_out, std::string cat) {

		PyObject *res = PyObject_CallMethod(object, "transform",
				"{s:l,s:l,s:l,s:l}", "shape_x", (long) n_cont_in_rows,
				"shape_y", (long) n_cont_in_cols, "data_in", (long) *(&cont_in),
				"data_out", (long) *(&cont_out));

		if (!res) {
			LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
								"Python transform method couldn't be called");
			PyErr_Print();
		} else {
			LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python transform method was called successfully");
		}
	}

private:
	PyObject *object;
};

#endif /* NVTABULAR_H_ */
