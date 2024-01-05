# RT-Octree: Accelerate PlenOctree Rendering with Batched Regular Tracking and Neural Denoising for Real-time Neural Radiance Fields (SIGGRAPH Asia 2023)

### [Project page](https://rt-octree.github.io/) | [Paper](https://doi.org/10.1145/3610548.3618214) | [arXiv](https://sjtueducn-my.sharepoint.com/:b:/g/personal/zixi_shu_sjtu_edu_cn/Eam50UugV1tJkwXKPFkbfF0BjN8p7KaACOGOQGJeb_INGw?e=ltDCyd) 


## Environment (Tested)
- Ubuntu 20.04
- Python 3.9
- CUDA 11.x
- Pytorch 1.11.0+cu113
- Libtorch 1.11.0

## Install
### Conda Environment
```bash
conda create -n RTOctree python=3.9
conda activate RTOctree
pip install -r requirements.txt
```

### Libtorch
Please unzip libtorch to the root of the repository.
```bash
wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip
```
You can download libtorch from these links:
- Linux: 
https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
- Windows: 
https://download.pytorch.org/libtorch/cu113/libtorch-win-shared-with-deps-1.11.0%2Bcu113.zip


## Data
For inference, we use pretrained PlenOctree models download from [here](https://drive.google.com/drive/folders/1oUxS1Why1NaCd-ioPR3UCbCLYpfrOacm). 

For training our GuidanceNet, **NeRF-Synthetic dataset** ([Download Link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1))  and **TanksAndTemple dataset** ([Download Link](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)) are used. The noisy input of these datasets can be rendered by disable the denoiser in renderer.

We also provide a preprocessed noisy dataset and trained model for each scene using `SPP=6` ([Download Link](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zixi_shu_sjtu_edu_cn/ErlevBCwkRxKnVf_W49JD2kBnn2XtGQcCocsMid2wdBqxQ?e=2DTbv3))
## Renderer

### Build
#### Linux
```bash
mkdir build
cd build
cmake ../renderer
make -j12
```
#### Windows
```bash
mkdir build
cd build
cmake ../renderer
```
Then find the `.sln` file in the `build` directory and use Visual Studio to build the project.

### Run
Use `volrend_headless` to perform offscreen rendering. Here is an example.
```bash
export DATASET=blender
export SCENE=lego
export TREE=../data/nerf_synthetic/$SCENE/tree.npz
export POSES=../data/nerf_synthetic/$SCENE/transforms_test.json
export TS_MODULE=../data/nerf_synthetic/$SCENE/ts_latest.ts
export OUT_DIR=../logs/$SCENE/test
export OPTIONS=../renderer/options/opt.json

# Test and write images
./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET -o $OUT_DIR
# Test FPS only
./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET
```
For Tanks and Temples dataset, `POSES` is point to the directory containing `*.txt` poses, eg. `export POSES=../data/TanksAndTemple/$SCENE/pose`; For LLFF dataset, `POSES` is point to the `poses_bounds.npy` file, eg. `export POSES=../data/nerf_llff_data/$SCENE/poses_bounds.npy`

You can also run an ImGui window using `volrend`, for example:
```bash
./volrend $TREE --ts_module=$TS_MODULE
```

## GuidanceNet Training
You need to prepare noisy images and place it under the same directory as `$POSES`. 
For example, there should be a `../data/nerf_synthetic/lego/spp_6` directory containing the noisy data for `spp=6`.

For example, you can render the noisy input using the following command:
```bash
export DATASET=blender
export SCENE=lego
export TREE=../data/nerf_synthetic/$SCENE/tree.npz
export POSES=../data/nerf_synthetic/$SCENE/transforms_test.json
export TS_MODULE=../data/nerf_synthetic/$SCENE/ts_latest.ts
export OUT_DIR=../data/nerf_synthetic/$SCENE/spp_6/test
export OPTIONS=../renderer/options/opt.json

# Write noisy buffers
./volrend_headless $TREE $POSES --options=$OPTIONS --ts_module=$TS_MODULE --dataset=$DATASET -o $OUT_DIR --write_buffer
```

Then you can train the denoiser by
```bash
python -m denoiser.main --config=denoiser/configs/blender.txt --task=train
```
The training settings is defined in `denoiser/configs/*.txt`.
After training, the trained model will be saved as torchscript module to `ts_<epoch>.ts` files.


## Reference
- Our initial code was borrowed from [PlenOctree Renderer](https://github.com/sxyu/volrend)

## Citation

If you find this code helpful for your research, please cite:

```
@inproceedings{shu2023rtoctree,
  title={RT-Octree: Accelerate PlenOctree Rendering with Batched Regular Tracking and Neural Denoising for Real-time Neural Radiance Fields},
  author={Shu, Zixi and Yi, Ran and Meng, Yuqi and Wu, Yutong and Ma, Lizhuang},
  booktitle={SIGGRAPH Asia 2023 Conference Papers},
  pages={1--11},
  year={2023}
}
```