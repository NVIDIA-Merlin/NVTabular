# Documentation

This folder contains the scripts necessary to build NVTabular's documentation.
You can view the generated [NVTabular documentation here](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).

# Contributing to Docs

Follow the instructions below to be able to build the docs.

## Steps to follow:
1. In order to build the docs, we need to install NVTabular in a conda env. [See installation instructions](https://github.com/NVIDIA/NVTabular).

2. Install required documentation tools and extensions:

`pip install sphinx recommonmark nbsphinx sphinx_rtd_theme`

Once NVTabular is installed, navigate to ../NVTabular/docs/. If you have your documentation written and want to turn it into HTML, run makefile:

#be in the same directory as your Makefile

`make html`

This should run Sphinx in your shell, and outputs to build/html/index.html.

View docs web page by opening HTML in browser:
First navigate to /build/html/ folder, i.e., cd build/html and then run the following command:

`python -m http.server`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://<host IP-Address>:8000`

Now you can check if your docs edits formatted correctly, and read well.

