#!/bin/bash
set -e
source activate rapids
apt-get update
apt install screen openjdk-8-jdk -y
conda install snakeviz bottleneck nodejs jupyterlab lightgbm bokeh=1.4.0 -y
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
pip install pytest pytest-cov sphinx jupyterlab-nvdashboard fastai==1.0.61 dask-cuda py-spy==0.4.0.dev0 tensorflow-gpu==2.3.0 grpcio==1.24.3 
conda install -c conda-forge jupytext black flake8 isort -y
jupyter labextension install jupyterlab-nvdashboard
cd /nvtabular;  TF_MEMORY_ALLOCATION="0.1" pytest tests/unit/
rm -rf /nvtabular/categories /nvtabular/dask-worker-space /nvtabular/ds_export /nvtabular/gb_categories
echo "cd /" >> /root/.bashrc
# cd /apex/; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Install HugeCTR
git clone -b v2.2.1-integration https://gitlab-master.nvidia.com/zehuanw/hugectr.git && \
    cd hugectr && \
    git submodule update --init --recursive && \
    sed -i '27,28 s/^/#/' ./test/utest/CMakeLists.txt && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="60;70" .. && \
    make -j && \
    mkdir /usr/local/hugectr && \
    make install && \
    chmod +x /usr/local/hugectr/bin/* && \
    rm -rf HugeCTR 
