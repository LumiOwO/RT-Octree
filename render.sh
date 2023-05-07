#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1 

export DATASET=blender
export TREE=../data/nerf_synthetic/lego/tree.npz
export POSES=../data/nerf_synthetic/lego/transforms_train.json
export OUT_DIR=../data/nerf_synthetic/lego/spp_6/train
export TS_MODULE=/root/volrend/logs/lego/vibrant-sun-8/ts_002000.ts
export OPTIONS=../renderer/options/blender.json

# export DATASET=tt
# export TREE=../data/TanksAndTemple/Truck/tree.npz
# export POSES=../data/TanksAndTemple/Truck
# export OUT_DIR=../data/TanksAndTemple/Truck/spp_6
# export TS_MODULE=/root/volrend/logs/lego/vibrant-sun-8/ts_002000.ts
# export OPTIONS=../renderer/options/blender.json

export FPS_ONLY=true

cd build

if [ "$FPS_ONLY" = true ]
then
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET
else
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET \
    -o $OUT_DIR --write_buffer
fi
