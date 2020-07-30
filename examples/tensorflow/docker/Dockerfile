# dev decides whether to copy notebooks in and
# run as root. Root is useful for cupti profiling
ARG dev=false
ARG pyver=3.7
FROM rapidsai/rapidsai-nightly:0.15-cuda10.1-base-ubuntu18.04-py${pyver} AS nvtabular

# configure environment
ENV HOME=/home/docker
WORKDIR $HOME
VOLUME $HOME
EXPOSE 8888 6006

# build example-specific libs
ADD . nvtabular/
ADD examples/tensorflow/docker/requirements.txt requirements.txt
RUN echo "nvtabular/." >>  requirements.txt && \
      source activate rapids && \
      conda install -c conda-forge -c rapidsai jupyterlab dask-cudf && \
      pip install -U --no-cache-dir -r requirements.txt && \
      rm -rf nvtabular requirements.txt

# set up jupyter notebook, including password token
# I would prefer to do the node installation with the
# previous conda install, but I'm having some issues
# when I try to do that
ARG token=nvidia
RUN source activate rapids && \
      conda install nodejs>=10.0.0 && \
      jupyter nbextension enable --py widgetsnbextension && \
      jupyter labextension install @jupyter-widgets/jupyterlab-manager

# configure CUDA libs
# ENV LD_LIBRARY_PATH="/usr/local/cuda-10.1/extras/CUPTI/lib64/:/usr/local/cuda-10.1/compat/:${LD_LIBRARY_PATH}"

# different images for dev and production
FROM nvtabular as true
ENTRYPOINT source activate rapids && jupyter lab --ip=0.0.0.0 --LabApp.token=${token}

FROM nvtabular as false
COPY examples/ $HOME
ENTRYPOINT source activate rapids && jupyter lab --ip=0.0.0.0 --LabApp.token=${token} --allow-root

FROM ${dev}
