## [NVTabular](https://github.com/NVIDIA/NVTabular) | NVTabular Inference API documentation

NVIDIA Merlin framework accelerates the recommendation pipeline end-2-end. As critical step of this pipeline, model deployment of ML/DL models is the process of making production ready models available in production environments, where they can provide predictions for new streaming data.

Here, we describe how to run the [Triton Inference Server](https://github.com/triton-inference-server/server) backend for Python to be able to do model deployment. The goal of [Python backend](https://github.com/triton-inference-server/python_backend) is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.

We provide an example notebook, `movielens_inference_example`, and explain the steps to do inference with Merlin Inference API.

# Getting Started 

In order to use Merlin Inference API, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTAbular and training a model. The other one is for serving/inferencing. 

## 1. Pulling the NVTabular Docker Container:

We start with pulling NVTabular container. This is to do preprocessing, feature engineering on our datasets, and then to train a DL model with PyT, TF or HugeCTR frameworks with processed datasets.

Before starting docker continer, first create a `nvt_triton` directory and `models` subdirectory on your host machine:

```
mkdir -p nvt_triton/models/
cd /nvt_triton
```
We will mount this directory into the NVTabular docker container.

NVTabular is available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

The beta (0.4) container is currently available. You can pull the container by running the following command:

```
docker run --gpus=all -it -v ${PWD}:/working_dir/ -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.4 /bin/bash
```
The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:

```
root@2efa5b50b909:
```

Activate the rapids conda environment by running the following command:
```
root@2efa5b50b909: source activate rapids
```
You should receive the following response, indicating that he environment has been activated:

```
(rapids)root@2efa5b50b909:
```
1) Install Triton Python Client Library:

You need to install tritonclient library to be able to run `movielens_deployment_example` notebook, and send request to the triton server. 

```
pip install nvidia-pyindex
pip install tritonclient
pip install geventhttpclient
```

2) Start the jupyter-lab server by running the following command:
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Open any browser to access the jupyter-lab server using `https://<host IP-Address>:8888`.

## 2. Run example notebooks:

There are two example notebooks that should be run in orders. The first one shows how to
- do preprocessing with NVTabular
- serialize and save a workflow to load later to transform new dataset
- train a TF MLP model and save it in the `/models` directory.

The following notebook shows `movielens_deployment_example` how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can open `movielens_example` and `movielens_deployment_example` notebooks. Note that you need to save your workflow and DL model in the `models` directory before launching the `tritonserver` as defined below. Then you can run the `movielens_deployment_example` notebook.

## 3. Build and Run the Triton Inference Server container:

1) Navigate to the `nvt_triton` directory that you saved NVTabular workflow and DL model on your host machine.
```
cd <path to nvt_triton>
```

2) Run the Triton Inference Server container.

docker run -it --name tritonserver --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/working_dir/ merlin/nvtabular:triton 

The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```
3) Your saved model should be in the `/models` directory. Navigate to the `models` working directory inside the triton server container to check the saved model:
```
cd /models
```
4) Start the triton server and run Triton with the example model repository you just created. 
```
tritonserver --model-repository `pwd`/models &
```
Once the models are successfully loaded, you can run the `movielens_deployment_example` notebook to send requests to the Triton IS.
