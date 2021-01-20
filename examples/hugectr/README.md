# NVTabular-HugeCTR Examples
## Run container
```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.3 /bin/bash

```

### Run Jupyter lab
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```