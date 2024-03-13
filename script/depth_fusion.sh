#!/usr/bin/env bash

DATASET="./DTU_TEST" #set to your DTU_TEST dataset direction
ROOT_DIR="./checkpoints/unfavorable" #set to your output direction

python depth_fusion.py --dataset DTU --geo_mask_thres 4 --full_fusion \
--dataset_dir=$DATASET --root_dir=$ROOT_DIR $@
