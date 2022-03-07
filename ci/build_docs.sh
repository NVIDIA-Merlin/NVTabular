#!/bin/bash
set -e
cd /var/jenkins_home/workspace/$JOB_NAME/nvtabular/
git checkout main

pip uninstall nvtabular -y
pip install . 

sphinx-multiversion docs/source docs/build/html/
cp -R docs/build/html/ ..
ls -1 | xargs rm -rf 
rm -rf .github .gitlab-ci.yml .gitignore .pre-commit-config.yaml
cp -R ../html/* .
cp main/.nojekyll .
git add --all
git commit -m "docs push $GIT_COMMIT"