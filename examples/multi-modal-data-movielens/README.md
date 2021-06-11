# Training Recommender Systems with Multi-Modal Data 

This project demonstrates how to build a recommender system from heterogeneous, multi-modal data sources (tabular, text, images...).

We will start from the movie-lens 25M data set, enriching it with image (movie poster) and text (movie sypnosis from IMDB). We will then employ domain-specific feature extractors, namely ResNet-50 and BART to extract features from images and texts. Finally, the multi-modal data is fed to the network in the form of pretrained embeddings for the task of predicting user-movie rating scores.


## Docker image
Unless otherwise stated, the notebooks should be executed from within the `merlin-tensorflow-training:0.5.2` docker container.

```
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow-training:0.5.2
docker run --gpus=all -it --rm --net=host --ipc=host  -v ${PWD}:/workspace nvcr.io/nvidia/merlin/merlin-tensorflow-training:0.5.2 bash

```
Then, from within the container, start Jupyter:

```
cd /workspace
source activate merlin
pip install jupyter jupyterlab
jupyter server extension disable nbclassic
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='admin'
```

The notebooks should be executed in the following order.

- [01-Download-Convert.ipynb](01-Download-Convert.ipynb)

- [02-Data-Enrichment.ipynb](02-Data-Enrichment.ipynb)

- [03-Feature-Extraction-Poster.ipynb](03-Feature-Extraction-Poster.ipynb): this notebook is executed using a ResNet container. See details in the notebook.

- [04-Feature-Extraction-Text.ipynb](04-Feature-Extraction-Text.ipynb): this notebook is executed using a HuggingFace NLP container. See details in the notebook.

- [05-Create-Feature-Store.ipynb](05-Create-Feature-Store.ipynb)

- [06a-Training-with-TF-with-pretrained-embeddings.ipynb](06a-Training-with-TF-with-pretrained-embeddings.ipynb)

- [06b-Training-wide-and-deep-with-pretrained-embedding.ipynb](06b-Training-wide-and-deep-with-pretrained-embedding.ipynb)
