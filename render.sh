#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1 

export DATASET=llff
export SCENE=horns

export FPS_ONLY=false
# export BUFFER=
export BUFFER=--write_buffer

if [ "$DATASET" = "blender" ]; then
   export TREE=../data/nerf_synthetic/$SCENE/tree.npz
   export POSES=../data/nerf_synthetic/$SCENE/transforms_train.json
   export OUT_DIR=../data/nerf_synthetic/$SCENE/spp_6/train
   export TS_MODULE=/root/volrend/logs/lego/vibrant-sun-8/ts_002000.ts
   export OPTIONS=../renderer/options/blender.json
elif [ "$DATASET" = "tt" ]; then
   export TREE=../data/TanksAndTemple/$SCENE/tree.npz
   export POSES=../data/TanksAndTemple/$SCENE/pose
   export OUT_DIR=../data/TanksAndTemple/$SCENE/spp_6
   export TS_MODULE=/root/volrend/logs/lego/vibrant-sun-8/ts_002000.ts
   export OPTIONS=../renderer/options/blender.json
elif [ "$DATASET" = "llff" ]; then
   export TREE=../data/nerf_llff_data/$SCENE/tree.npz
   export POSES=../data/nerf_llff_data/$SCENE/poses_bounds.npy
   export OUT_DIR=../data/nerf_llff_data/$SCENE/spp_6
   export TS_MODULE=/root/volrend/logs/lego/vibrant-sun-8/ts_002000.ts
   export OPTIONS=../renderer/options/blender.json
else
   echo "Invalid dataset type."
fi


cd build

if [ "$FPS_ONLY" = true ]
then
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET
else
   ./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET \
    -o $OUT_DIR $BUFFER
fi
