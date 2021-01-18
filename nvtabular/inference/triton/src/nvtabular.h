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
//using namespace py::literals;

//#include <Python.h>

/*
 class NVTabular {
 public:

 NVTabular() {
 py::initialize_interpreter();
 //py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
 //nt = nvtabular();
 //py::module_ sys = py::module_::import("numpy");
 //Py_Initialize();

 pName = PyUnicode_DecodeFSDefault("nvt");
 pModule = PyImport_Import(pName);

 if (pModule != NULL) {
 pFunc = PyObject_GetAttrString(pModule, "nvtabular");
 } else {
 PyErr_Print();
 fprintf(stderr, "Failed to load \"%s\"\n", "nvt");
 }

 printf("\n******** initialized\n");
 }
 ~NVTabular() {

 //Py_DECREF(pName);
 //Py_DECREF(pModule);
 //Py_DECREF(pDict);
 //Py_DECREF(pFunc);
 //Py_DECREF(pValue);

 //Py_Finalize();
 py::finalize_interpreter();
 printf("\n******** finalized\n");
 }

 void Deserialize(std::string path) {
 //nt.attr("deserialize")("path");
 }

 void Transform(float cont, std::string cat) {
 //nt.attr("transform")(cont, cat);
 }

 private:
 //py::object nt;
 PyObject *pName, *pModule, *pDict, *pFunc, *pValue;
 };
 */

class NVTabular {
public:

	NVTabular() {
		py::initialize_interpreter();
		printf("\nPython interpreter is initialized\n");
		LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python interpreter is initialized");
	}
	~NVTabular() {
		py::finalize_interpreter();
		LOG_MESSAGE(TRITONSERVER_LOG_INFO,
				"Python interpreter is  finalized\n");
	}

	void Deserialize(std::string path) {
		py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		py::object nt = nvtabular();
		nt.attr("deserialize")("path");
	}

	void Transform(float* cont_in, int n_cont_in_rows, int n_cont_in_cols,
			float* cont_out, std::string cat) {
		py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		py::object nt = nvtabular();

		py::dict cont_in_cai;
		std::tuple<long, long> shape_in((long) n_cont_in_rows,
				(long) n_cont_in_cols);
		cont_in_cai["shape"] = shape_in;
		std::tuple<long, bool> data_in((long) *(&cont_in), false);
		cont_in_cai["data"] = data_in;

		py::dict cont_out_cai;
		std::tuple<long, long> shape_out((long) n_cont_in_rows,
				(long) n_cont_in_cols);
		cont_out_cai["shape"] = shape_out;
		std::tuple<long, bool> data_out((long) *(&cont_out), false);
		cont_out_cai["data"] = data_out;

		nt.attr("transform")(cont_in_cai, cont_out_cai, cat);
	}

};

#endif /* NVTABULAR_H_ */
