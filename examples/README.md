# NVTabular Example Notebooks

We have a collection of Jupyter notebooks that are based on different datasets.
These example notebooks demonstrate how to use NVTabular with TensorFlow, PyTorch, and [HugeCTR](https://github.com/NVIDIA/HugeCTR).
Each example provides additional information about NVTabular's features.

If you'd like to create a full conda environment to run the example notebooks, do the following:

1. Use the [environment files](https://github.com/NVIDIA/NVTabular/tree/main/conda/environments) that have been provided to install the CUDA Toolkit (11.0 or 11.2).
2. Clone the NVTabular repo and run the following commands from the root directory:
   ```bash
   conda env create -f=conda/environments/nvtabular_dev_cuda11.2.yml
   conda activate nvtabular_dev_11.2
   python -m ipykernel install --user --name=nvt
   pip install -e .
   jupyter notebook
   ```
   When opening a notebook, be sure to select `nvt` from the `Kernel->Change Kernel` menu.

## Structure

The example notebooks are structured as follows and should be reviewed in this order:

- 01-Download-Convert.ipynb: Demonstrates how to download the dataset and convert it into the correct format so that it can be consumed.
- 02-ETL-with-NVTabular.ipynb: Demonstrates how to execute the preprocessing and feature engineering pipeline (ETL) with NVTabular on the GPU.
- 03-Training-with-TF.ipynb: Demonstrates how to train a model with TensorFlow based on the ETL output.
- 03-Training-with-PyTorch.ipynb: Demonstrates how to train a model with PyTorch based on the ETL output.
- 03-Training-with-HugeCTR.ipynb: Demonstrates how to train a model with HugeCTR based on the ETL output.

## Available Example Notebooks

### 1. [Getting Started with MovieLens](https://github.com/NVIDIA/NVTabular/tree/main/examples/getting-started-movielens)

The MovieLens25M is a popular dataset for recommender systems and is used in academic publications. Most users are familiar with this dataset, so this example notebook is focusing primarily on the basic concepts of NVTabular, which includes:

- Learning NVTabular with NVTabular's high-level API
- Using single-hot/multi-hot categorical input features with NVTabular
- Using the NVTabular dataloader with the TensorFlow Keras model
- Using the NVTabular dataloader with PyTorch

### 2. [Advanced Ops with Outbrain](https://github.com/NVIDIA/NVTabular/tree/main/examples/advanced-ops-outbrain)

The [Outbrain dataset](https://www.kaggle.com/c/outbrain-click-prediction) is based on a Kaggle Competition in which Kagglers were challenged to predict which ads and other forms of sponsored content that their global users would click. This example notebook demonstrates how to use the available NVTabular operators, write a custom operator, and train a Wide&Deep model with the NVTabular dataloader in TensorFlow.

### 3. [Scaling Large Datasets with Criteo](https://github.com/NVIDIA/NVTabular/tree/main/examples/scaling-criteo)

[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) provides the largest publicly available dataset for recommender systems with a size of 1TB of uncompressed click logs that contain 4 billion examples. This example notebook demonstrates how to scale NVTabular, use multiple GPUs and multiple nodes with NVTabular for ETL, and train a recommender system model with the NVTabular dataloader for PyTorch.

### 4. [Multi-GPU with MovieLens](https://github.com/NVIDIA/NVTabular/tree/main/examples/multi-gpu-movielens)

In the Getting Started with MovieLens example, we explain the fundamentals of NVTabular and its dataloader, HugeCTR, and Triton Inference. With this example, we revisit the same dataset but demonstrate how to perform multi-GPU training with the NVTabular dataloader in TensorFlow.

### 5. [Winning Solution of the RecSys2020 Competition](https://github.com/NVIDIA/NVTabular/tree/main/examples/winning-solution-recsys2020-twitter)

Twitter provided a dataset for the [RecSys2020 challenge](http://www.recsyschallenge.com/2020/). The goal was to predict user engagement based on 200M user-tweet pairs. This example notebook demonstrates how to use NVTabular's available operators for feature engineering and train a XGBoost model on the GPU with dask.

### 6. [Applying the Techniques to other Tabular Problems with Rossmann](https://github.com/NVIDIA/NVTabular/tree/main/examples/tabular-data-rossmann)

Rossmann operates over 3,000 drug stores across seven European countries. Historical sales data for 1,115 Rossmann stores are provided. The goal is to forecast the **Sales** column for the test set. Kaggle hosted it as a [competition](https://www.kaggle.com/c/rossmann-store-sales/overview).

## Running the Example Notebooks

You can run the example notebooks by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries.
Alternatively, Docker containers are available from the NVIDIA GPU Cloud (NGC) at <http://ngc.nvidia.com/catalog/containers/> with pre-installed versions.
Depending on which example you want to run, you should use any one of these Docker containers:

- `merlin-hugectr` (contains NVTabular with HugeCTR)
- `merlin-tensorflow` (contains NVTabular with TensorFlow)
- `merlin-pytorch` (contains NVTabular with PyTorch)

Beginning with the 22.06 release, each container includes the software for training models and performing inference.

To run the example notebooks using Docker containers, do the following:

1. Pull the container by running the following command:

   ```sh
   docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host <docker container> /bin/bash
   ```

   **NOTES**:

   - If you are running Getting Started with MovieLens, Advanced Ops with Outbrain, or the Tabular Problems with Rossmann example notebooks, add a `-v ${PWD}:/root/` argument to the preceding Docker command.
   The `PWD` environment variable refers to a local directory on your computer, and you should specify this same directory and with the `-v` argument when you run a container to perform inference.
   Follow the instructions for starting Triton Inference Server that are provided in the inference notebooks.
   - If you are running `Training-with-HugeCTR` notebooks, please add `--cap-add SYS_NICE` to the `docker run` command to suppress the `set_mempolicy: Operation not permitted` warnings.

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```sh
   root@2efa5b50b909:
   ```

1. Start the jupyter-lab server by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   ```shell
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

1. After you log in to JupyterLab, navigate to the `/nvtabular` directory to try out the example notebooks.
