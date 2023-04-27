#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1 

export TREE=../../nerf_synthetic/lego/tree.npz
export POSES=../../nerf_synthetic/lego/transforms_test.json
export OUT_DIR=../logs/temp
export TS_MODULE=../logs/lego/ts_000100.ts

cd build

./volrend_headless $TREE $POSES --ts_module=$TS_MODULE \
    -o $OUT_DIR --write_buffer