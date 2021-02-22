ETL
===

You can run NVTabular on the cloud using: 
* [Amazon Web Services (AWS)](amazon-web-services)
* [Google Cloud Platform (GCP)](google-cloud-platform)

## Amazon Web Services

Amazon Web Services (AWS) offers [EC2 instances with NVIDIA GPU support](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing). NVTabular can be used with 1x, 4x, or 8x GPU instances (or multi-node setup). We're using an EC2 instance with 8x NVIDIA A100 GPUs to demonstrate the steps below. Please check out the $/h for this instance type and adjust the type. 

To run NVTabular on the cloud using AWS, do the following:

1. Start the AWS EC2 instance with the [NVIDIA Deep Learning AMI image](https://aws.amazon.com/marketplace/pp/NVIDIA-NVIDIA-Deep-Learning-AMI/B076K31M1S) using the aws-cli.

   ```
   # Starts the P4D instance with 8x NVIDIA A100 GPUs (take a look at the $/h for this instance type before using them)
   aws ec2 run-instances --image-id ami-04c0416d6bd8e4b1f --count 1 --instance-type p4d.24xlarge --key-name <MyKeyPair> --security-groups <my-sg>
   ```

2. SSH into the machine.
    
3. Create a RAID volume by running the following command:

   Depending on the EC2 instance, the machine may include local disk storage. We can optimize the performance by creating a [RAID volume](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/raid-config.html). Based on our experience, two NVME volumes yield the best performance.

   ```
   sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/nvme1n1 /dev/nvme2n1

   sudo mkfs.ext4 -L MY_RAID /dev/md0
   sudo mkdir -p /mnt/raid
   sudo mount LABEL=MY_RAID /mnt/raid

   sudo chmod -R 777 /mnt/raid

   # Copy dataset inside raid directory:
   cp -r data/ /mnt/raid/data/
   ```

4. Launch the NVTabular docker container by running the 

   ```
   docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE -v /mnt/raid:/raid nvcr.io/nvidia/nvtabular:0.3 /bin/bash
   ```

5. Start the jupyter-lab server by running the following command:
    
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```

## Google Cloud Platform

The Google Cloud Platform offers [Compute Engine instances with NVIDIA GPU support](https://cloud.google.com/compute/docs/gpus). We're using a VM with 8x NVIDIA A100 GPUs and eight local SSD-NVMe devices configured as RAID 0 to demonstrate the steps below.

To run NVTabular on the cloud using GCP, do the following:

1. Configure and create the VM.
    * **GPU**: 8xA100 (a2-highgpu-8g)
    * **Boot Disk**: Ubuntu version 18.04
    * **Storage**: Local 8xSSD-NVMe

2. [Install the appropriate NVIDIA drivers and CUDA](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#ubuntu-driver-steps).
   ```
   curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
   sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
   sudo apt -y update
   sudo apt -y install cuda
   nvidia-smi # Check installation
   ```
   
3. [Install Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
   ```
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get -y update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi # Check Installation
   ```

4. Configure the storage as RAID 0 by running the following commands:
   ```
   sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/nvme0n1 /dev/nvme0n2
   sudo mkfs.ext4 -L MY_RAID /dev/md0
   sudo mkdir -p /mnt/raid
   sudo mount LABEL=MY_RAID /mnt/raid
   sudo chmod -R 777 /mnt/raid

   # Copy data to RAID
   cp -r data/ /mnt/raid/data/
   ```

5. Run the container using the following command:
   ```
   docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE -v /mnt/raid:/raid nvcr.io/nvidia/nvtabular:0.3 /bin/bash
   ```
