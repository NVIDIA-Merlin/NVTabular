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
using namespace py::literals;

class NVTabular {
public:

	NVTabular() {
		py::initialize_interpreter();
		//py::object nvtabular = py::module_::import("nvt").attr("nvtabular");
		//nt = nvtabular();
	}
	~NVTabular() {
		py::finalize_interpreter();
	}

	void Deserialize(std::string path) {
		//nt.attr("deserialize")("path");
	}

	void Transform(float cont, std::string cat) {
		//nt.attr("transform")(cont, cat);
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
private:
	//py::object nt;
};

#endif /* NVTABULAR_H_ */
