# NVTabular with cudf gds

## Build container
From NVTabular root directory
```
bash examples/gds/docker/create_container.sh
```
## Run container
```
docker run --runtime=nvidia --rm -it \
    -v /path/to/data:/data\
    -p 9999:8888 -p 9797:8787 -p 9796:8786 \
    gds_rel /bin/bash 
```
