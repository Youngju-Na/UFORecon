DATASET=/home/yourname/dataset/DTU
LOG_DIR=./logs/dtu

python main.py --max_epochs 16 --stage 1 --batch_size 1 --lr 0.0001 \
--weight_rgb 1.0 --weight_depth 1.0 --weight_perceptual 1.0 \
--patch_size 48 --view_selection_type best \
--train_ray_num 1024 --volume_reso 96 \
--root_dir=$DATASET --logdir=$LOG_DIR $@ \

