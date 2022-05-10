# Training and Inference with HugeCTR Model

In this folder, we provide two example notebooks, [Training-with-HugeCTR](https://github.com/NVIDIA/NVTabular/blob/main/examples/getting-started-movielens/inference-HugeCTR/Training-with-HugeCTR.ipynb) and [Triton-Inference-with-HugeCTR](https://github.com/NVIDIA/NVTabular/blob/main/examples/getting-started-movielens/inference-HugeCTR/Triton-Inference-with-HugeCTR.ipynb), and explain the steps to do inference with Merlin Inference API after training a model with HugeCTR framework. 

## Getting Started 

There are two containers that are needed in order to use the Merlin Inference API. The first one is for preprocessing with NVTabular and training a model with HugeCTR framework. The other one is for serving/inference. 

## 1. Pull the Merlin Training Docker Container:

We start with pulling the `Merlin-Training` container. This is to do preprocessing, feature engineering on our datasets using NVTabular, and then to train a DL model with HugeCTR framework with processed datasets.

Before starting docker container, first create a `nvt_triton` directory and `data` subdirectory on your host machine:

```
mkdir -p nvt_triton/data/
cd nvt_triton
```
We will mount `nvt_triton` directory into the NVTabular docker container.

Merlin containers are available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

You can pull the `Merlin-Training` container by running the following command:

```
docker run --gpus=all -it -v ${PWD}:/model/ -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host nvcr.io/nvidia/merlin/merlin-training:21.11 /bin/bash
```

The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:


```
root@2efa5b50b909:
```

1) Start the jupyter-lab server by running the following command. In case the container does not have `JupyterLab`, you can easily [install](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) it using pip.
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Open any browser to access the jupyter-lab server using `https://<host IP-Address>:8888`.

## 2. Run example notebooks:

There are two example notebooks that should be run in order. The first one [Training-with-HugeCTR](https://github.com/NVIDIA/NVTabular/blob/main/examples/getting-started-movielens/inference-HugeCTR/Training-with-HugeCTR.ipynb) shows how to
- do preprocessing with NVTabular
- serialize and save a workflow to load later to transform new dataset
- train a HugeCTR MLP model and save it in the `/models` directory.

The following notebook [Triton-Inference-with-HugeCTR](https://github.com/NVIDIA/NVTabular/blob/main/examples/getting-started-movielens/inference-HugeCTR/Triton-Inference-with-HugeCTR.ipynb) shows how to send request to Triton IS 
- to transform new data with NVTabular
- to generate prediction results for new dataset.

Now you can start `Training-with-HugeCTR` and `Triton-Inference-with-HugeCTR` notebooks. Note that you need to mount the directory where your NVTAbular workflow and HugeCTR model is saved when launching the `tritonserver` docker image as shown below. Then you can run the `Triton-Inference-with-HugeCTR` example notebook once the server is started.

## 3. Build and Run the Triton Inference Server container:

1) Navigate to the `nvt_triton` directory that you saved NVTabular workflow, Tensorflow model and the ensemble model.
```
cd <path to nvt_triton>
```

2) Launch Merlin Triton Inference Server container:
```
docker run -it --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/model/ nvcr.io/nvidia/merlin/merlin-inference:21.11

```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

3) Your saved models should be in the `/model` directory. Navigate to the `model` working directory inside the triton server container to check the saved models:
```
cd /model
```
4) Start the triton server and run Triton with the example model repository you just created. Note that you need to provide correct path for the models directory, and `movielens.json` file.
```
tritonserver --model-repository=/model/models/ --backend-config=hugectr,movielens=/model/models/movielens/1/movielens.json --backend-config=hugectr,supportlonglong=true --model-control-mode=explicit
```

After you start Triton you will see output on the console showing the server starting up. At this stage it does not load any model, you will load the `movielens_ens` model in the  `Triton-Inference-with-HugeCTR` notebook to be able to send the request. All the models should load successfully. If a model fails to load the status will report the failure and a reason for the failure. Once the models are successfully loaded, you can send requests to the Triton IS. 
