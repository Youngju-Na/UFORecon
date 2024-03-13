#!/usr/bin/env bash

ROOT_DIR="/home/yourname/UFORecon/checkpoints/favorable"
# ROOT_DIR=$1
# voxel_size matter: 1.5 in DTU dataset, 0.005 in BlendedMVS dataset
python tsdf_fusion.py --n_view 3 --voxel_size 1.5 --test_view 23 24 33 --dataset dtu --root_dir=$ROOT_DIR @