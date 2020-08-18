# dev decides whether to copy notebooks in and
# run as root. Root is useful for cupti profiling
ARG dev=false
FROM nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04 AS base

# install python and cudf
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && \
      /conda/bin/conda update -n base conda && \
      /conda/bin/conda create --name nvtabular -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
        -c defaults cudf=0.15 python=3.7 cudatoolkit=10.1 dask-cudf pip nodejs>=10.0.0

# set up shell so we can do "source activate"
ENV PATH=${PATH}:/conda/bin
SHELL ["/bin/bash", "-c"]

# set up nvtabular and example-specific libs
ADD . nvtabular/
ADD examples/tensorflow/docker/requirements.txt requirements.txt
RUN source activate nvtabular && \
      echo "nvtabular/." >>  requirements.txt && \
      pip install -U --no-cache-dir -r requirements.txt && \
      rm -rf nvtabular requirements.txt

# configure environment
ENV HOME=/home/docker
WORKDIR $HOME
VOLUME $HOME
EXPOSE 8888 6006

# configure jupyter notebook
# add arg for login token and enable tensorboard
ARG token=nvidia
RUN source activate nvtabular && \
      jupyter nbextension enable --py widgetsnbextension && \
      jupyter labextension install @jupyter-widgets/jupyterlab-manager

# add cupti to ld library path for profiling
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/extras/CUPTI/lib64/"

# different images for dev and production
FROM base AS true
ENTRYPOINT source activate nvtabular && jupyter lab --ip=0.0.0.0 --LabApp.token=${token}

FROM base AS false
COPY examples/tensorflow/ $HOME
ENTRYPOINT source activate nvtabular && jupyter lab --ip=0.0.0.0 --LabApp.token=${token} --allow-root

FROM ${dev}
