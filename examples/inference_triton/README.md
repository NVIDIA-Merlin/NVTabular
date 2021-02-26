## Merlin Inference API

NVIDIA Merlin framework accelerates the recommendation pipeline end-2-end. As critical step of this pipeline, model deployment of machine learning (ML)/deep learning (DL) models is the process of making production ready models available in production environments, where they can provide predictions for new data.

Here, we describe how to run the [Triton Inference Server](https://github.com/triton-inference-server/server) backend for Python to be able deploy a model. The goal of the [Python backend](https://github.com/triton-inference-server/python_backend) is to let you serve models written in Python by Triton Inference Server (IS) without having to write any C++ code. 

We provide two example notebooks, [movielens_TF](https://github.com/NVIDIA/NVTabular/blob/main/examples/inference_triton/movielens-TF.ipynb) and [movielens_inference](https://github.com/NVIDIA/NVTabular/blob/main/examples/inference_triton/movielens_inference.ipynb), and explain the steps to do inference with Merlin Inference API. 

Merlin Inference API is designed for a seamless integration between NVTabular and Triton IS, where Triton simplifies the deployment of AI models at scale in
production. This integration makes online data transformation and generation of the query results for new data pretty easy. Users use `NVTabular` to preprocess training data and collect statistics (in our example for categorical features) at the training phase on training data. After this transformation and fit process, NVTabular’s workflow object is serialized to be loaded at inference phase to transform the new dataset with the collected statistics. By deserializing the saved NVTabular workflow, these statistics are applied to transform the new dataset at inference phase: whenever a new query is received by the inference server, the data is first fed into NVTabular to call the transform of the NVTabular workflow object. The output is then passed to the ML/DL model (a Tensorflow model in our example) and prediction results for the query are generated accordingly.


# Getting Started 

In order to use Merlin Inference API, there are two containers that the user needs to build and launch. The first one is for preprocessing with NVTabular and training a model. The other one is for serving/inference. 

## 1. Pulling the NVTabular Docker Container:

We start with pulling NVTabular container. This is to do preprocessing, feature engineering on our datasets, and then to train a DL model with PyT, TF or HugeCTR frameworks with processed datasets.

Before starting docker continer, first create a `nvt_triton` directory and `data` subdirectory on your host machine:

```
mkdir -p nvt_triton/data/
cd nvt_triton
```
We will mount `nvt_triton` directory into the NVTabular docker container.

Merlin containers are available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

You can pull the `Merlin-Tensorflow-Training` container by running the following command:

```
docker run --gpus=all -it -v ${PWD}:/working_dir/ -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/merlin_tf_training /bin/bash
```
The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:

```
root@2efa5b50b909:
```

Activate the rapids conda environment by running the following command:
```
root@2efa5b50b909: source activate rapids
```
You should receive the following response, indicating that the environment has been activated:

```
(rapids)root@2efa5b50b909:
```
1) Install Triton Python Client Library:

You need the Triton Python Client library to be able to run `movielens_inference` notebook, and send request to the triton server. In case triton client library is missing, you can install with the following commands:

```
pip install nvidia-pyindex
pip install tritonclient
pip install geventhttpclient
```
Additionally, you might need to install `unzip` and `graphviz` packages if they are missing. You can do that with the following commands:

```
apt-get update
apt-get install unzip
pip install graphviz 
```

2) Start the jupyter-lab server by running the following command. In case the container does not have `JupyterLab`, you can easily [install](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) it either using conda or pip.
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Open any browser to access the jupyter-lab server using `https://<host IP-Address>:8888`.

## 2. Run example notebooks:

There are two example notebooks that should be run in order. The first one [movielens_TF](https://github.com/NVIDIA/NVTabular/blob/main/examples/inference_triton/movielens-TF.ipynb) shows how to
- do preprocessing with NVTabular
- serialize and save a workflow to load later to transform new dataset
- train a TF MLP model and save it in the `/models` directory.

The following notebook [movielens_inference](https://github.com/NVIDIA/NVTabular/blob/main/examples/inference_triton/movielens_inference.ipynb) shows how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can start `movielens_TF` and `movielens_inference` notebooks. Note that you need to save your workflow and DL model in the `models` directory before launching the `tritonserver` as defined below. Then you can run the `movielens_inference` example notebook once the server is started.

## 3. Build and Run the Triton Inference Server container:

1) Navigate to the `nvt_triton` directory that you saved NVTabular workflow, Tensorflow model and the ensemble model.
```
cd <path to nvt_triton>
```

2) Launch Merlin Triton Inference Server container:
```
docker run -it --name tritonserver --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/working_dir/ nvcr.io/nvidia/merlin_inference
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

Activate the rapids conda environment by running the following command:
```
root@02d56ff0738f:/opt/tritonserver#  source activate rapids
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

All the models should show "READY" status to indicate that they are loaded correctly. If a model fails to load the status will report the failure and a reason for the failure. If your model is not displayed in the table, check the path to the model repository and your CUDA drivers.

Once the models are successfully loaded, you can run the `movielens_inference` notebook to send requests to the Triton IS. Note that, by default Triton will not start if models are not loaded successfully.
