#! /usr/bin/env bash

## 1. Install torkit3d and apex
bash scripts/clone-or-pull.sh https://github.com/zyc00/Point-SAM third_party/Point-SAM

git submodule update --init third_party/Point-SAM/third_party/torkit3d
FORCE_CUDA=1 pip install --no-build-isolation third_party/torkit3d

git submodule update --init third_party/Point-SAM/third_party/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" third_party/apex

## 2. Install PointOps
bash scripts/clone-or-pull.sh https://github.com/Pointcept/SAMPart3D third_party/SAMPart3D

pushd third_party/SAMPart3D/libs/pointops
python setup.py install
popd

# spconv-cu124 already installed with pixi
pip install flash-attn --no-build-isolation
