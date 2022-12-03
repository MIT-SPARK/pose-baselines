#cd ../

python -W ignore train.py \
--outfile logs/shapenet_depth/chair/ \
--dataset_type shapenet_depth \
--object chair \
--data_type synthetic

