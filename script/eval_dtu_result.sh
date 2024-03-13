#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0"

DATASET="datasets/SampleSet" ##set to your SampleSet_MVS_Data
OUT_DIR='checkpoints/unfavorable_1_16_36'
python evaluation/dtu_eval.py --dataset_dir $DATASET $@ --outdir $OUT_DIR $@

python evaluation/log_to_csv.py --log_file_path $OUT_DIR/eval_final.log --csv_output_path $OUT_DIR/out.csv