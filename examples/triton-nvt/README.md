## [NVTabular](https://github.com/NVIDIA/NVTabular) | NVTabular Inference API documentation

NVIDIA Merlin framework accelerates the recommendation pipeline end-2-end. As important part of this pipeline, model deployment of ML/DL models is the process of making production ready models available in production environments, where they can provide predictions for new streaming data.

Here, we describe how to run the [Triton Inference Server](https://github.com/triton-inference-server/server) backend for Python to be able to do model deployment. The goal of [Python backend](https://github.com/triton-inference-server/python_backend) is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.

We provide an example notebook, `movielens_inference_example`, and explain the steps to do inference with Merlin Inference API.

# Getting Started 

In order to use Merlin Inference API, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTAbular and training a model. The other one is for serving/inferencing. 

## 1. Pulling the NVTabular Docker Container:

We start with pulling NVTabular container. This is to do preprocessing, feature engineering on our dataset, and then to train a DL model with PyT, TF or HugeCTR frameworks.

For instructions how pull and launch NVTabular docker container, we refer the user to the [Getting Started](https://github.com/NVIDIA/NVTabular/blob/main/README.md#getting-started) guide of NVTabular README document).

## 2. Build and Run the Triton Inference Server container:

1) Create a models directory on your host machine
```
mkdir -p models
cd models
```

3) Run the Triton Inference Server container.

docker run -it --name tritonserver --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/models:/models merlin/nvtabular:triton 

The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```
4 ) Navigate to the models directory inside the container:
cd /models
