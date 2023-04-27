#!/bin/bash
set -e

export CONFIG=blender
export TASK=train

python -m denoiser.main --config=denoiser/configs/$CONFIG.txt --task=$TASK