## [NVTabular](https://github.com/NVIDIA/NVTabular) | [Merlin Inference API documentation]

NVIDIA Merlin accelerates the recommendation pipeline end-2-end. 

Here, we describe how to run the Triton backend for Python. The goal of Python backend is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.
We provide an example notebook, `movielens_inference_example`, and explain the steps to do inference with Merlin Inference API.

# Getting Started 

In order to use Merlin Inference API, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTAbular and training a model. The other one is for serving/inference. 

## Pulling the PP Container:
The Triton backend for Python.

Run the Triton Inference Server container.

1) Create a models directory on your host machine

mkdir -p models
cd models

2) Clone triton inference server python backend github repo.

git clone https://github.com/triton-inference-server/python_backend.git

3) Run the Triton Inference Server container.

docker run --gpus all -v ${PWD}:/working_dir --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:20.12-py3

Navigate to the working_dir
cd /working_dir

Install miniconda inside the docker container. Refer to this documentation for miniconda installation.
After installing miniconda, create a conda environment and install rapids inside this environment.
#conda create --name
