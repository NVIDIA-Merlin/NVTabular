# NVTabular Example Notebooks

In this library, we provide a collection of Jupyter notebooks, which demonstrates the functionality of NVTabular.

## Inventory

- [Getting Started with NVTabular](01-Getting-started.ipynb): Get started with NVTabular by processing data on the GPU.

- [Advanced NVTabular workflow](02-Advanced-NVTabular-workflow.ipynb): Understand NVTabular in more detail by defining more advanced workflows and learn about different operators

- [Running on multiple GPUs or on CPU](03-Running-on-multiple-GPUs-or-on-CPU.ipynb): Run NVTabular in different environments, such as multi-GPU or CPU-only mode.

In addition, NVTabular is used in many of our examples in other Merlin libraries. You can explore more complex processing pipelines in following examples:
- [End-To-End Examples with Merlin](https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples)
- [Training Examples with Merlin Models](https://github.com/NVIDIA-Merlin/models/tree/main/examples)
- [Training Examples with Transformer4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples)

## Running the Example Notebooks

You can run the example notebooks by [installing NVTabular](https://github.com/NVIDIA/NVTabular#installation) and other required libraries. Alternatively, Docker containers are available from the NVIDIA GPU Cloud (NGC) at <http://ngc.nvidia.com/catalog/containers/> with pre-installed versions.
Depending on which example you want to run, you should use any one of these Docker containers:

- `merlin-hugectr` (contains NVTabular with HugeCTR)
- `merlin-tensorflow` (contains NVTabular with TensorFlow)
- `merlin-pytorch` (contains NVTabular with PyTorch)

Beginning with the 22.06 release, each container includes the software for training models and performing inference.

To run the example notebooks using Docker containers, perform the following steps:

1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     <docker container> /bin/bash
   ```

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

1. Start the JupyterLab server by running the following command:

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

## Troubleshooting

If you experience any trouble running the example notebooks, check the latest [troubleshooting](https://nvidia-merlin.github.io/NVTabular/main/resources/troubleshooting.html) documentation.
