#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
TEST_DIR="datasets/DTU_TEST" ##set to your DTU_TEST dataset direction 
EXP_DIR="checkpoints/unfavorable_1_16_36/mesh" ##set to your output direction
python evaluation/clean_mesh.py --root_dir $TEST_DIR $@ --out_dir $EXP_DIR $@ --n_view 5 --set 0 --test_ref_view 1 16 36 


