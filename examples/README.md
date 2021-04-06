# NVTabular Examples

We created a collection of jupyter notebooks based on different datasets. The examples cover how to use NVTabular in combination with TensorFlow, PyTorch and [HugeCTR](https://github.com/NVIDIA/HugeCTR). Each example explains some features of NVTabular in detail. We provide an order to go through the notebooks as the later ones expect some prior knowledge.

## Structure

Each example are structured in multiple notebooks:
- 01-Download-Convert.ipynb: Instruction to download the dataset and convert it into the correct format to consume it in the next notebooks
- 02-ETL-with-NVTabular.ipynb: Execute preprocessing and feature engineering pipeline (ETL) with **NVTabular** on GPU
- 03a-Training-with-TF.ipynb: Training a model with **TensorFlow** based on the ETL output
- 03b-Training-with-PyTorch.ipynb: Training a model with **PyTorch** based on the ETL output
- 03c-Training-with-HugeCTR.ipynb: Training a model with **HugeCTR** based on the ETL output
- 04 - containing a range of additional notebooks for one dataset.

## Examples

### 1. [Getting Started with MovieLens](https://github.com/NVIDIA/NVTabular/tree/main/examples/getting-started-movielens)

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications. Most users are familiar with the dataset and we will teach the **basic concepts of NVTabular**:
- Learning **NVTabular** for using GPU-accelerated ETL (Preprocess and Feature Engineering)
- Getting familiar with **NVTabular's high-level API**
- Using single-hot/multi-hot categorical input features with **NVTabular**
- Using **NVTabular dataloader** with TensorFlow Keras model
- Using **NVTabular dataloader** with PyTorch

### 2. [Advanced Ops with Outbrain](https://github.com/NVIDIA/NVTabular/tree/main/examples/advanced-ops-outbrain)

[Outbrain dataset](https://www.kaggle.com/c/outbrain-click-prediction) is based on a Kaggle Competition in which Kagglers were challenged to predict on which ads and other forms of sponsored content its global users would click. We will teach to **use more of the available NVTabular operators**:
- Getting familiar with a wide range of NVTabular operators
- Writing a custom operator
- Training Wide&Deep model with NVTabular dataloader in TensorFlow

### 3. [Scaling to large Datasets with Criteo](https://github.com/NVIDIA/NVTabular/tree/main/examples/scaling-criteo)

[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) provides the largest publicly available dataset for recommender systems, having a size of 1TB uncompressed click logs of 4 billion examples. We will teach to **scale NVTabular**:
- Using **multiple GPUs and multiple nodes** with NVTabular for ETL
- Training recommender system model with NVTabular dataloader for PyTorch

### 4. [Winning Solution of the RecSys2020 Competition](https://github.com/NVIDIA/NVTabular/tree/main/examples/winning-solution-recsys2020-twitter)

Twitter provided a dataset for the [RecSys2020 challenge](http://www.recsyschallenge.com/2020/). The task was to predict user engagement based on 200M user-tweet pairs. We will show another example using NVTabular:
- Using more available NVTabular operators for Feature Engineering
- Training a XGBoost model on GPU with dask

### 5. [Applying the Techniques to other Tabular Problems with Rossmann](https://github.com/NVIDIA/NVTabular/tree/main/examples/tabular-data-rossmann) 

Rossmann operates over 3,000 drug stores in 7 European countries. Historical sales data for 1,115 Rossmann stores are provided. The task is to forecast the "Sales" column for the test set. Kaggle hosted it as a [competition](https://www.kaggle.com/c/rossmann-store-sales/overview).
- Using NVTabular for sales prediction

## Start Examples

You can run the examples by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternativly, docker conatiners are available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions. Depending which example you want to run, you should select the docker container:
- Merlin-Tensorflow-Training contains NVTabular with TensorFlow
- Merlin-Pytorch-Training contains NVTabular with PyTorch
- Merlin-Training contains NVTabular with HugeCTR
- Merlin-Inference contains NVTabular with TensorFlow and Triton Inference support

### Start Examples with Docker Container

You can pull the container by running the following command:

```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE <docker container> /bin/bash
```

**NOTE**: If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.

The container will open a shell when the run command execution is completed. You'll have to start jupyter lab on the Docker container. It should look similar to this:
```
root@2efa5b50b909:
```

1. Activate the ```merlin``` conda environment by running the following command:
   ```
   root@2efa5b50b909: source activate merlin
   ```

   You should receive the following response, indicating that the environment has been activated:
   ```
   (merlin)root@2efa5b50b909:
   ```
2. Install jupyter-lab with `conda` or `pip`: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

   ```
   pip install jupyterlab
   ```

3. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

4. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

5. Once in the server, navigate to the ```/nvtabular/``` directory and try out the examples.


