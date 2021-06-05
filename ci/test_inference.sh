#!/bin/bash

cd /nvtabular/
git pull origin main

tritonserver --model-repository /model/models --backend-config=tensorflow,version=2 --model-control-mode=explicit &
sleep 1m

pytest tests/integration/test_notebooks.py::test_tf_inference_multihot_examples