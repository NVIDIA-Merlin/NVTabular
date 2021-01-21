## [NVTabular](https://github.com/NVIDIA/NVTabular) | [API documentation](https://nvidia.github.io/NVTabular/main/index.html)


[NVTabular](https://github.com/NVIDIA/NVTabular), a component of [**NVIDIA Merlin Open Beta**](https://developer.nvidia.com/nvidia-merlin), is a feature engineering and preprocessing library for tabular data that is designed to quickly and easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) library.

Recommender systems require massive datasets to train, particularly for deep learning based solutions. The transformation of these datasets in order to prepare them for model training is particularly challenging. Often the time taken to complete the necessary steps, such as feature engineering, categorical encoding, and normalization of continuous variables exceeds the time it takes to train a model.

With NVTabular being a part of the Merlin ecosystem, it also works with the other Merlin components including [HugeCTR](https://github.com/NVIDIA/HugeCTR) and [Triton Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) to provide end-to-end acceleration of recommender systems on the GPU.

NVTabular is designed to support data scientists and machine learning (ML) engineers train (deep learning) recommender systems and resolve tabular data problems by allowing them to:

* prepare datasets quickly and easily for experimentation so that more models can be trained
* work with datasets that exceed GPU and CPU memory without having to worry about scale.
* focus on what to do with the data, and not how to do it, using our abstraction at the operation level.

It is also meant to help ML/Ops Engineers with deploying models into production by providing faster dataset transformation. This makes it easy for production models to be trained more frequently and kept up to date, helping improve responsiveness and model performance.

The library is designed to be interoperable with both PyTorch and TensorFlow using dataloaders that we have developed as extensions of native framework code. NVTabular provides the option to shuffle data during preprocessing, allowing the dataloader to load large contiguous chunks from files rather than individual elements. This allows us to do per epoch shuffles orders of magnitude faster than a full shuffle of the dataset. Loading tabular data is almost always a training bottleneck. In our benchmarking, we've seen 10x improvements in training time on the GPU relative to the native dataloaders.

Extending beyond model training, we plan on integrating with model-serving frameworks like NVIDIA’s Triton Inference Server. This will create a clear path to production inference for these models and allow the feature engineering and preprocessing steps performed on the data during training to be automatically applied to incoming data during inference.

Our goal is faster iteration on massive tabular datasets, both for experimentation during training, and also for production model responsiveness.

### Getting Started
NVTabular is available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

The beta (0.3) container is currently available. You can pull the container by running the following command:

```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.3 /bin/bash
```

If you are running on Docker version 19 and higher, change ```--runtime=nvidia``` to ```--gpus all```.

The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container.
It should look similar to this:
```
root@2efa5b50b909:
```

1. Activate the ```rapids``` conda environment by running the following command:
```
root@2efa5b50b909: source activate rapids
```

You should receive the following response, indicating that he environment has been activated:
```
(rapids)root@2efa5b50b909:
```

2. Start the jupyter-lab server by running the following command:
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

3. Open any browser to access the jupyter-lab server using <MachineIP>:8888.

Once in the server, navigate to the ```/nvtabular/``` directory and explore the code base or try out some of the examples.

The container contains the codebase along with all of our dependencies, particularly [RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) and a range of [examples](./examples). The easiest way to get started is to simply launch the container above and explore the examples within it.

The code base and related examples can be found at the following directory location within the container:
```
/nvtabular/
```

#### Conda

NVTabular can be installed with Anaconda from the ```nvidia``` channel:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=10.2
```

#### Amazon Web Services (AWS)

Amazon Web Services (AWS) offers [EC2 instances with NVIDIA GPU support](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing). NVTabular can be used with 1x, 4x or 8x GPU instances (or multi-node setup). This example uses an EC2 instance with 8x NVIDIA A100 GPUs. Please checkout the $/h for this instance type and adjust the type. 

1. Start AWS EC2 instance with [NVIDIA Deep Learning AMI image](https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S) using aws-cli.

```
# Starts P4D instance with 8x NVIDIA A100 GPUs (take a look on the $/h for this instance type before using them)
aws ec2 run-instances --image-id ami-04c0416d6bd8e4b1f --count 1 --instance-type p4d.24xlarge --key-name <MyKeyPair> --security-groups <my-sg>
```

2. SSH into the machine
    
3. Create RAID volume

Depending on the EC2 instance, the machine includes local disk storage. We can optimize the performance by creating a [RAID volume](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/raid-config.html). In our experience, 2 NVME volumes yield best performance.

```
sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/nvme1n1 /dev/nvme2n1

sudo mkfs.ext4 -L MY_RAID /dev/md0
sudo mkdir -p /mnt/raid
sudo mount LABEL=MY_RAID /mnt/raid

sudo chmod -R 777 /mnt/raid

# Copy dataset inside raid directory:
cp -r data/ /mnt/raid/data/
```

4. Launch NVTabular docker container

```
docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE -v /mnt/raid:/raid nvcr.io/nvidia/nvtabular:0.3 /bin/bash
```

5. Start the jupyter-lab server by running the following command:
    
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```


#### Google Cloud Platform (GCP)

Google Cloud Platform (GCP) offers [Compute Engine instances with NVIDIA GPU Support](https://cloud.google.com/compute/docs/gpus). This example uses a VM with 8x NVIDIA A100 GPUs, and 8 local SSD-NVMe devices configured as RAID 0.
1. Configure and create the VM. 
    * GPU: 8xA100 (a2-highgpu-8g)
    * Boot disk: Ubuntu 18.04
    * Storage: Local 8xSSD-NVMe

2. [Install Nvidia Drivers and CUDA](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#ubuntu-driver-steps)
```
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt -y update
sudo apt -y install cuda
nvidia-smi # Check installation
```
3. [Install Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi # Check Installation
```

4. Configure Storage as RAID 0
```
sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/nvme0n1 /dev/nvme0n2
sudo mkfs.ext4 -L MY_RAID /dev/md0
sudo mkdir -p /mnt/raid
sudo mount LABEL=MY_RAID /mnt/raid
sudo chmod -R 777 /mnt/raid

# Copy data to RAID
cp -r data/ /mnt/raid/data/
```

5. Run container
```
docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE -v /mnt/raid:/raid nvcr.io/nvidia/nvtabular:0.3 /bin/bash
```

### Examples and Tutorials

The following use cases can be found in our [API documentation examples section](https://nvidia.github.io/NVTabular/main/examples/index.html).
* Preprocessing
* Feature engineering
* Dataloading in Tensorflow, PyTorch, and HugeCTR

Performance of the Criteo DRLM workflow demonstrates the effectiveness of the NVTabular library. The original ETL script provided in Numpy took over five days to complete. Combined with CPU training, the total iteration time is over one week. By optimizing the ETL code in spark and running on a DGX-1 equivalent cluster, we were able to bring that time down to three hours for ETL and one hour for training.

With NVTabular running on a single V100 32GB GPU, we were able to complete ETL in 13 minutes. With a DGX-1 cluster of eight V100 GPUs, we can accelerate ETL to 3 minutes. Combined with [HugeCTR](http://www.github.com/NVIDIA/HugeCTR/), we can process the dataset and train the full model in only 6 minutes. This fast iteration is the goal of NVTabular and the [Merlin application framework](https://developer.nvidia.com/nvidia-merlin). We're working on A100 benchmarks and will share them as soon as they are available.

We also have a [simple tutorial](examples/rossmann/) that demonstrates similar functionality on a much smaller dataset. A pipeline for the [Rossman store sales dataset](https://www.kaggle.com/c/rossmann-store-sales) that feeds into a [fast.ai tabular data model](https://docs.fast.ai/tabular.learner.html) is provided.

### Contributing

If you'd like to contribute to the library directly, please see [Contributing.md](./CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [this survey](https://developer.nvidia.com/merlin-devzone-survey).

### Learn More

If you're interested in learning more about how NVTabular works under the hood, we've provided this [more detailed description of the core functionality](HowItWorks.md).

We also have [API documentation](https://nvidia.github.io/NVTabular/main/index.html) that outlines the specifics of the available calls within the library.

You can find public presentations and blog posts under [Resources](Resources.md).
