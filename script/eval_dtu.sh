#!/usr/bin/env bash


DATASET="/home/yourname/3d-recon/datasets/DTU_TEST_uforecon/DTU_TEST" #set to your DTU_TEST dataset direction

LOAD_CKPT="./pretrained/uforecon.ckpt"  #set to your checkpoint direction

OUT_DIR="./outputs_1_16_36" #set to your output direction

python main.py --extract_geometry --set 0 \
--test_n_view 3 --test_ray_num 400 --volume_reso 96 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
