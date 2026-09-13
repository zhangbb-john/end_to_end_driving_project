#!/bin/bash

# Check if huggingface-hub is installed
if ! python -m pip show huggingface-hub > /dev/null 2>&1; then
  echo "huggingface-hub is not installed. Installing now..."
  python -m pip install huggingface-hub
else
  echo "huggingface-hub is already installed."
fi

mkdir Bench2Drive-mini

#hf download --resume-download --repo-type dataset rethinklab/Bench2Drive --include "StaticCutIn_Town15_Route26405_Weather0.tar.gz" --local-dir Bench2Drive-mini --local-dir-use-symlinks False
hf download --repo-type dataset rethinklab/Bench2Drive --include "StaticCutIn_Town15_Route26405_Weather0.tar.gz" --local-dir Bench2Drive-mini

