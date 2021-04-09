# Inference with Tensorflow Model

Here, we describe how to run the [Triton Inference Server](https://github.com/triton-inference-server/server) backend for Python to be able deploy a Tensorflow (TF) model. The goal of the [Python backend](https://github.com/triton-inference-server/python_backend) is to let you serve models written in Python by Triton Inference Server (IS) without having to write any C++ code. 

We provide four example notebooks, [movielens-TF](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-TF.ipynb), [movielens-inference](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-inference.ipynb), [movielens-multihot-TF](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-multihot-TF.ipynb) and [movielens-multihot-inference](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-multihot-inference.ipynb), and explain the steps to do inference with Merlin Inference API. 

## Getting Started 

In order to use Merlin Inference API with TF framework, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTabular and training a model with TF. The other one is for serving/inference. 

## 1. Pull the Merlin Tensorflow Training Docker Container:

We start with pulling NVTabular container. This is to do preprocessing, feature engineering on our datasets using NVTabular, and then to train a DL model with TF framework with processed datasets.

Before starting docker container, first create a `nvt_triton` directory and `data` subdirectory on your host machine:

```
mkdir -p nvt_triton/data/
cd nvt_triton
```

We will mount `nvt_triton` directory into the `Merlin-Tensorflow-Training` docker container. Merlin containers are available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

You can pull the `Merlin-Tensorflow-Training` container by running the following command:

```
docker run --gpus=all -it -v ${PWD}:/model/ -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host nvcr.io/nvidia/merlin/merlin-tensorflow-training:0.4 /bin/bash
```
The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the docker container. It should look similar to this:

```
root@2efa5b50b909:
```

Activate the merlin conda environment by running the following command:
```
root@2efa5b50b909: source activate merlin
```
You should receive the following response, indicating that the environment has been activated:

```
(merlin)root@2efa5b50b909:
```
1) Start the jupyter-lab server by running the following command. In case the container does not have `JupyterLab`, you can easily [install](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) it either using conda or pip.
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Open any browser to access the jupyter-lab server using `https://<host IP-Address>:8888`.

## 2. Run Single-hot example notebooks:

There are two example notebooks that should be run in order. The first one [movielens_TF](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-TF.ipynb) shows how to
- do preprocessing with NVTabular
- serialize and save a workflow to load later to transform new dataset
- train a TF MLP model and save it in the `/models` directory.


The following notebook [movielens-inference](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-inference.ipynb) shows how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can start `movielens-TF` and `movielens-inference` notebooks. Note that you need to save your workflow and DL model in the `models` directory before launching the `tritonserver` as defined below. Then you can run the `movielens-inference` example notebook once the server is started.

## 3. Run Multi-hot example notebooks:

There are two example notebooks that should be run in order. The first one [movielens-multihot-TF](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-multihot-TF.ipynb) shows how to
- do preprocessing a dataset with multi-hot categorical columns with NVTabular
- train a TF MLP model and save three models `movielens_mh`, `movielens_mh_nvt`, `movielens_mh_tf` in the `/models-multihot` directory.

The following notebook [movielens-multihot-inference](https://github.com/NVIDIA/NVTabular/tree/main/examples/inference_triton/inference-TF/movielens-multihot-inference.ipynb) shows how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can start `movielens-multihot-TF` and `movielens-multihot-inference` notebooks. Note that you need to save your workflow and DL model in the `models_multihot` directory before launching the `tritonserver` as defined below. Then you can run the `movielens-multihot-inference` example notebook once the server is started.


## 4. Build and Run the Triton Inference Server container:

1) Navigate to the `nvt_triton` directory that you saved NVTabular workflow, Tensorflow model and the ensemble model.
```
cd <path to nvt_triton>
```

2) Launch Merlin Triton Inference Server container:

```
docker run -it --name tritonserver --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/model/ nvcr.io/nvidia/merlin/merlin-inference:0.5
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

Activate the rapids conda environment by running the following command:
```
root@02d56ff0738f:/opt/tritonserver#  source activate merlin
```

3) Your saved model should be in the `/models` directory for single-hot example or in the `/models_multihot` directory for multi-hot example. Navigate to the `model` working directory inside the triton server container to check the saved models:
```
cd /model
```
4) Start the triton server and run Triton with the example model repository you just created. Note that you need to provide correct path for the `models` directory.
```
tritonserver --model-repository /model/models/ --backend-config=tensorflow,version=2
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

All the models should show "READY" status to indicate that they are loaded correctly. If a model fails to load the status will report the failure and a reason for the failure. If your model is not displayed in the table, check the path to the model repository and your CUDA drivers.

Once the models are successfully loaded, you can run the `movielens-inference` notebook to send requests to the Triton IS. Note that, by default Triton will not start if models are not loaded successfully. 

You can repeat all the substeps under the step 4 by replacing `models` with `models_multihot` folder to send request for the `multihot` example.
