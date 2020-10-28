# HugeCTR configuration
CMAKE_BUILD_TYPE=Release
SM="60;70;80"
VAL_MODE=OFF
ENABLE_MULTINODES=OFF
NCCL_A2A=ON

# Install HugeCTR
git clone --depth 1 --branch "v2.2.1" https://github.com/NVIDIA/HugeCTR &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    sed -i '27,28 s/^/#/' ./test/utest/CMakeLists.txt && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DSM=$SM .. && \
    make -j && \
    mkdir /usr/local/hugectr && \
    make install && \
    chmod +x /usr/local/hugectr/bin/* && \
    rm -rf HugeCTR 

PATH=/usr/local/hugectr/bin:$PATH 
