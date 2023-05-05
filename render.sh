#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1 

# export DATASET=blender
# export TREE=../data/nerf_synthetic/lego/tree.npz
# export POSES=../data/nerf_synthetic/lego/transforms_test.json
# export OUT_DIR=../data/nerf_synthetic/lego/spp_4/test
# export TS_MODULE=../logs/ts_latest.ts
# export OPTIONS=../renderer/options/blender.json

export DATASET=tt
export TREE=../data/TanksAndTemple/Barn/tree.npz
export POSES=../data/TanksAndTemple/Barn
export OUT_DIR=../data/TanksAndTemple/Barn/spp_4
export TS_MODULE=../logs/ts_latest.ts
export OPTIONS=../renderer/options/blender.json

export FPS_ONLY=false

cd build

if [ "$FPS_ONLY" = true ]
then
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET
else
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET \
    -o $OUT_DIR --write_buffer
fi
