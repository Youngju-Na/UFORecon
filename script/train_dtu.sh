DATASET=dataset/DTU
LOG_DIR=checkpoints/favorable

# UFORecon-trans-resume
python main.py --max_epochs 16 --batch_size 1 --weight_rgb 1.0 --weight_depth 1.0 --train_ray_num 1024 \
    --volume_type correlation --volume_reso 96 \
    --logdir=$LOG_DIR --view_selection_type best --train_n_view 5 \
    --test_n_view 3 --test_ref_view 1 16 36 --mvs_depth_guide 1 \
    --load_ckpt pretrained/uforecon.ckpt --depth_pos_encoding --explicit_similarity --root_dir=$DATASET $@ \

