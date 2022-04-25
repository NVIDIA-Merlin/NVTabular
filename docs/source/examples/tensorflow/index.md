# Accelerating TensorFlow Tabular Workflows with NVTabular
## TODO: Include data section?
Get Criteo, run preproc notebook, mount volume, make sure you have space for tfrecords, etc.

## Build container
From root directory
```
docker build -t $USER/nvtabular-tf-example -f examples/tensorflow/docker/Dockerfile .
```
## Run container
```
docker run --rm -it \
    -v /path/to/data:/data -v /path/to/write/tfrecords:/tfrecords \
    -p 8888:8888 -p 6006:6006 \
    --gpus 1 $USER/nvtabular-tf-example
```
And navigate to `<your ip address>:8888/?token=nvidia`
