#!/usr/bin/env bash

DATASET="datasets/DTU_TEST_uforecon/DTU_TEST" #set to your DTU_TEST dataset direction
LOAD_CKPT="pretrained/uforecon.ckpt"  #set to your checkpoint direction
OUT_DIR="checkpoints/unfavorable_1_16_36" #set to your output direction

python main.py --extract_geometry --set 0 \
--volume_type "correlation" --volume_reso 96 --view_selection_type "best" \
--depth_pos_encoding --mvs_depth_guide 1 --explicit_similarity \
--test_n_view 3 --test_ray_num 800 --test_ref_view 1 16 36 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
