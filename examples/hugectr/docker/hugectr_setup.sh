#!/bin/bash

# HugeCTR configuration
CMAKE_BUILD_TYPE=Release
SM="60;70;80"
VAL_MODE=OFF
ENABLE_MULTINODES=OFF
NCCL_A2A=ON

# Install dependencies
apt-get update
apt install screen openjdk-8-jdk -y
conda install snakeviz bottleneck nodejs jupyterlab lightgbm bokeh=1.4.0 -y
conda install -c conda-forge nvtx
pip install torch pytest pytest-cov sphinx jupyterlab-nvdashboard fastai dask-cuda py-spy==0.4.0.dev0 tensorflow-gpu==2.3.0 grpcio==1.24.3 
conda install -c conda-forge jupytext black flake8 isort nvtx -y
jupyter labextension install jupyterlab-nvdashboard

# Install HugeCTR
git clone -b v2.3 https://github.com/NVIDIA/HugeCTR.git &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    sed -i '28,29 s/^/#/' ./test/utest/CMakeLists.txt && \
    sed -i '22 s/^/#/' ./tools/CMakeLists.txt && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DSM=$SM .. && \
    make -j && \
    mkdir /usr/local/hugectr && \
    make install && \
    chmod +x /usr/local/hugectr/bin/* && \
    chmod +x /usr/local/hugectr/lib/* &&\
    rm -rf HugeCTR 

export PATH=/usr/local/hugectr/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
