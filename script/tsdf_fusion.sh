#!/usr/bin/env bash

ROOT_DIR="/home/yourname/ssd/UFORecon/checkpoints_mvi06/0000868e/2_15_1_22"
# ROOT_DIR=$1
# voxel_size matter: 1.5 in DTU dataset, 0.005 in BlendedMVS dataset
python tsdf_fusion.py --n_view 4 --voxel_size 0.05 --test_view 1 2 15 22 --dataset mvimage --root_dir=$ROOT_DIR --test_scan 0000868e $@
