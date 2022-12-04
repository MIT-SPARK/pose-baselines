
cd ../../c3po/baselines

# script for training fcgf on real (depth) point cloud data of shapenet
python fcgf_train_shapenet.py \
--type 'real' \
--voxel_size 0.025 \
--out_dir "fcgf/data/output_shapenet/realv10" \
--model_n_out 16 \
--conv1_kernel_size 3 \
--train_data_len_shapenet 2048 \
--val_data_len_shapenet 512 \
--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"
