## [NVTabular](https://github.com/NVIDIA/NVTabular) | NVTabular Inference API documentation

NVIDIA Merlin framework accelerates the recommendation pipeline end-2-end. As critical step of this pipeline, model deployment of ML/DL models is the process of making production ready models available in production environments, where they can provide predictions for new streaming data.

Here, we describe how to run the [Triton Inference Server](https://github.com/triton-inference-server/server) backend for Python to be able to do model deployment. The goal of [Python backend](https://github.com/triton-inference-server/python_backend) is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.

We provide two example notebooks, `movielens_TF` and `movielens_deployment`, and explain the steps to do inference with Merlin Inference API.

# Getting Started 

In order to use Merlin Inference API, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTAbular and training a model. The other one is for serving/inferencing. 

## 1. Pulling the NVTabular Docker Container:

We start with pulling NVTabular container. This is to do preprocessing, feature engineering on our datasets, and then to train a DL model with PyT, TF or HugeCTR frameworks with processed datasets.

Before starting docker continer, first create a `nvt_triton` directory and `models` and `data` subdirectories on your host machine:

```
mkdir -p nvt_triton/models/
mkdir -p nvt_triton/data/
cd /nvt_triton
```
We will mount `nvt_triton` directory into the NVTabular docker container.

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

The following notebook shows `movielens_deployment` how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can open `movielens_TF` and `movielens_deployment` notebooks. Note that you need to save your workflow and DL model in the `models` directory before launching the `tritonserver` as defined below. Then you can run the `movielens_deployment` notebook once the server is started.

## 3. Build and Run the Triton Inference Server container:

1) Navigate to the `nvt_triton` directory that you saved NVTabular workflow and Tensorflow model.
```
cd <path to nvt_triton>
```

2) Launch Merlin Triton Inference Server container:
```
docker run -it --name tritonserver --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/working_dir/ merlin/nvt:triton 
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```
3) Your saved model should be in the `/models` directory. Navigate to the `models` working directory inside the triton server container to check the saved model:
```
cd /models
```
4) Start the triton server and run Triton with the example model repository you just created. Note that you need to provide correct path for the models directory.
```
tritonserver --model-repository /working_dir/models/
```

After you start Triton you will see output on the console showing the server starting up and loading the model. When you see output like the following, Triton is ready to accept inference requests.

```
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| movielens     | 1       | READY  |
| movielens_nvt | 1       | READY  |
| movielens_tf  | 1       | READY  |
+---------------+---------+--------+
...
...
I0216 15:28:24.026506 71 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0216 15:28:24.027067 71 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0216 15:28:24.068597 71 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

All the models should show "READY" status to indicate that they loaded correctly. If a model fails to load the status will report the failure and a reason for the failure. If your model is not displayed in the table check the path to the model repository and your CUDA drivers.

Once the models are successfully loaded, you can run the `movielens_deployment` notebook to send requests to the Triton IS.
