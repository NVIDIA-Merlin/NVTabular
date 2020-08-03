# HugeCTR configuration
CMAKE_BUILD_TYPE=Debug
SM=70
VAL_MODE=OFF
ENABLE_MULTINODES=OFF
NCCL_A2A=ON

git clone -b cgarg/parquet-v2.2-integ https://gitlab-master.nvidia.com/zehuanw/hugectr.git &&\
    cd hugectr && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DSM=$SM .. && \
    make -j && \
    mkdir /usr/local/hugectr && \
    make install && \
    chmod +x /usr/local/hugectr/bin/* && \
    rm -rf HugeCTR 

PATH=/usr/local/hugectr/bin:$PATH