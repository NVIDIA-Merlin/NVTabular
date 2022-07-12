#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as build_pybind11
from setuptools import find_namespace_packages, find_packages, setup
from setuptools.command.develop import develop as _develop

try:
    import versioneer
except ImportError:
    # we have a versioneer.py file living in the same directory as this file, but
    # if we're using pep 517/518 to build from pyproject.toml its not going to find it
    # https://github.com/python-versioneer/python-versioneer/issues/193#issue-408237852
    # make this work by adding this directory to the python path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import versioneer


class develop(_develop):
    def run(self):
        # running setup.py develop doesn't seem to run 'build_py' force this to run
        # so we get our proto files installed
        self.run_command("build_py")
        super().run()


ext_modules = [
    Pybind11Extension(
        "nvtabular_cpp",
        [
            "cpp/nvtabular/__init__.cc",
            "cpp/nvtabular/inference/__init__.cc",
            "cpp/nvtabular/inference/categorify.cc",
            "cpp/nvtabular/inference/fill.cc",
        ],
        define_macros=[("VERSION_INFO", versioneer.get_version())],
        include_dirs=["./cpp/"],
    ),
]


cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_pybind11
cmdclass["develop"] = develop


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements("./requirements.txt")

setup(
    name="nvtabular",
    version=versioneer.get_version(),
    packages=find_packages(include=["nvtabular*"]) + find_namespace_packages(include=["merlin*"]),
    url="https://github.com/NVIDIA-Merlin/NVTabular",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=install_reqs,
)
