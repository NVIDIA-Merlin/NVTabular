# HugeCTR configuration
CMAKE_BUILD_TYPE=Release
SM=70
VAL_MODE=OFF
ENABLE_MULTINODES=OFF
NCCL_A2A=ON

# Install packages
pip install jupyterlab 
pip install pynvml
pip install torch torchvision

# Install NVTabular
pip install -e /nvtabular/

# Install HugeCTR
git clone -b v2.2.1-integration https://gitlab-master.nvidia.com/zehuanw/hugectr.git &&\
    cd hugectr && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DSM=$SM .. && \
    make -j && \
    mkdir /usr/local/hugectr && \
    make install && \
    chmod +x /usr/local/hugectr/bin/* && \
    rm -rf HugeCTR 

export PATH=/usr/local/hugectr/bin:$PATH
