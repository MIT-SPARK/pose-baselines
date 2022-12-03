#cd ../

python -W ignore train.py \
--outfile logs/shapenet_full/all/ \
--dataset_type shapenet_full \
--object all \
--data_type synthetic \
--writer True


