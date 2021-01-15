/*
 * nvtabular.h
 *
 *  Created on: Jan 13, 2021
 *      Author: oyilmaz
 */

#ifndef NVTABULAR_H_
#define NVTABULAR_H_

#include <pybind11/embed.h>

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
		//py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		//nt = nvtabular();

		printf("\n******** initialized\n");
	}
	~NVTabular() {
		py::finalize_interpreter();
		printf("\n******** finalized\n");
	}

	void Deserialize(std::string path) {
		py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		py::object nt = nvtabular();
		nt.attr("deserialize")("path");
	}

	void Transform(float* cont_in, int n_cont_in, float* cont_out, int n_cont, std::string cat) {
		py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		py::object nt = nvtabular();

		py::dict cont_in_cai;
		std::tuple<long> shape_in((long) n_cont_in);
		cont_in_cai["shape"] = shape_in;
		std::tuple<long, bool> data_in((long) *(&cont_in), false);
		cont_in_cai["data"] = data_in;

		py::dict cont_out_cai;
		std::tuple<long> shape_out((long) n_cont_in);
		cont_out_cai["shape"] = shape_out;
		std::tuple<long, bool> data_out((long) *(&cont_out), false);
		cont_out_cai["data"] = data_out;

		nt.attr("transform")(cont_in_cai, cont_out_cai, cat);
	}

	/*
	 void Test() {
	 //auto va = py::module::import("numpy");
	 py::print("******* pybind_test called!");
	 auto kwargs = py::dict("name"_a="World", "number"_a=42);
	 auto message = "Hello, {name}! The answer is {number}"_s.format(**kwargs);
	 py::print(message);

	 py::module_ sys = py::module_::import("sys");
	 py::print(sys.attr("path"));
	 }*/

};

#endif /* NVTABULAR_H_ */
