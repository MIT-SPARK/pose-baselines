#cd ../

python -W ignore train.py \
--outfile logs/modelnet/ \
--dataset_path dataset/ModelNet/ \
--dataset_type modelnet \
--data_type synthetic \
--categoryfile dataset/modelnet40_half1.txt \
--writer True

