#!/bin/bash
set -e

# get the nvtabular directory
ci_directory="$(dirname -- "$(readlink -f -- "$0")")"
nvt_directory="$(dirname -- $ci_directory)"
cd $nvt_directory

echo "Installing NVTabular"
python -m pip install --user --upgrade pip setuptools wheel pybind11 numpy==1.20.3 setuptools==59.4.0
python -m pip uninstall nvtabular -y
python setup.py develop --user --no-deps

# following checks requirement requirements-dev.txt to be installed
echo "Running black --check"
black --check .
echo "Running flake8"
flake8 .
echo "Running isort"
isort -c .
echo "Running bandit"
bandit -q -ll --recursive nvtabular
echo "Running pylint"
pylint nvtabular tests bench
echo "Running flake8-nb"
flake8-nb examples

# check for broken symlinks (indicates a problem with the docs usually)
broken_symlinks=$(find . -xtype l)
if [ ! -z "$broken_symlinks" ]; then 
    echo "Error - found broken symlinks in repo:"
    echo $broken_symlinks | awk '{gsub(" ","\n\t "); print "\t",$0}'
    exit 1; 
fi

# build the docs, treating warnings as errors
echo "Building docs"
make -C docs html SPHINXOPTS="-W -q"

# test out our codebase
py.test -x -rsx --cov-config tests/unit/.coveragerc --cov-report term-missing --cov-report xml --cov-fail-under 70 --cov=. tests/unit/
