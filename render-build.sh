#!/usr/bin/env bash

# install git-lfs
apt-get update
apt-get install -y git-lfs

# enable lfs
git lfs install

# pull large files
git lfs pull

# install uv
pip install uv

# install dependencies from pyproject.toml
uv sync --frozen
