#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1 

export TREE=../../nerf_synthetic/lego/tree.npz
export POSES=../../nerf_synthetic/lego/transforms_test.json
# export OUT_DIR=../../nerf_synthetic/lego/spp_4/test
export OUT_DIR=../logs/temp
export TS_MODULE=../logs/lego/ts_000100.ts

export FPS_ONLY=true

cd build

if [ "$FPS_ONLY"=true ]
then
   ./volrend_headless $TREE $POSES --ts_module=$TS_MODULE
else
   ./volrend_headless $TREE $POSES --ts_module=$TS_MODULE \
    -o $OUT_DIR --write_buffer
fi
