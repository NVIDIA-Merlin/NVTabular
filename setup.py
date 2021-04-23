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
import subprocess
import sys
from distutils.spawn import find_executable

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

import versioneer


class build_proto(build_ext):
    def run(self):
        protoc = None
        if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
            protoc = os.environ["PROTOC"]
        else:
            protoc = find_executable("protoc")
        if protoc is None:
            sys.stderr.write("protoc not found")
            sys.exit(1)

        for source in ["nvtabular/inference/triton/model_config.proto"]:
            output = source.replace(".proto", "_pb2.py")

            if not os.path.exists(output) or (os.path.getmtime(source) > os.path.getmtime(output)):
                with open(output, "a") as src:
                    src.write("# flake8: noqa" + os.linesep)
                    src.write("# fmt: off" + os.linesep)
                subprocess.check_call([protoc, "--python_out=.", source])
                with open(output, "r+") as src:
                    new_src_content = (
                        "# flake8: noqa"
                        + os.linesep
                        + "# fmt: off"
                        + os.linesep
                        + src.read()
                        + "# fmt: on"
                        + os.linesep
                    )
                    src.seek(0)
                    src.write(new_src_content)

        # Run original build_ext command
        build_ext.run(self)


cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_proto


setup(
    name="nvtabular",
    version=versioneer.get_version(),
    packages=find_packages(),
    url="https://github.com/NVIDIA/NVTabular",
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
)
