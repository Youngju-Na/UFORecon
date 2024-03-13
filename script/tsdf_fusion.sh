#!/usr/bin/env bash

ROOT_DIR="checkpoints/unfavorable_1_16_36"
# ROOT_DIR=$1
# voxel_size matter: 1.5 in DTU dataset, 0.005 in BlendedMVS dataset
python tsdf_fusion.py --n_view 3 --voxel_size 1.5 --test_view 1 16 36 --dataset dtu --root_dir=$ROOT_DIR 