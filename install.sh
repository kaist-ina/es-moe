#!/bin/bash
set -e

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPT_PATH

# Get DeepSpeed
DEPS_CACHE_DIR=$HOME/.cache/esmoe/deps

mkdir -p $DEPS_CACHE_DIR
git -C $DEPS_CACHE_DIR/DeepSpeed pull origin v0.14.2 || git clone https://github.com/microsoft/DeepSpeed.git -b v0.14.2 $DEPS_CACHE_DIR/DeepSpeed
patch --directory=$DEPS_CACHE_DIR/DeepSpeed --strip=1 --forward --reject-file=- < deepspeed.patch || true

# Install dependencies
# pip install six regex git+https://github.com/NVIDIA/TransformerEngine.git@stable
pip install --editable $DEPS_CACHE_DIR/DeepSpeed megatron/core/segment_manager megatron/core/shared_pinned_memory .

# clean build directory
rm -rf megatron/core/segment_manager/build
rm -rf megatron/core/shared_pinned_memory/build
