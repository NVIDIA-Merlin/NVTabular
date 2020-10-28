# NVTabular-HugeCTR Examples

## Build container
From NVTabular root directory
```
bash examples/hugectr/docker/create_container.sh
```
## Run container
```
docker run --runtime=nvidia --rm -it \
    -v /path/to/data:/data\
    -p 9999:8888 -p 9797:8787 -p 9796:8786 \
    hugectr_rel /bin/bash 
```

### Run Jupyter lab
```
cd /nvt
kill $(pgrep jupyter-lab)
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```