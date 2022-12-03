#cd ../

python -W ignore train.py \
--outfile logs/shapenet_depth/all/ \
--dataset_type shapenet_depth \
--object all \
--data_type synthetic \
--writer True

