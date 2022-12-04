#cd ../

python -W ignore train.py \
--dataset_path dataset/ModelNet/ \
--dataset_type modelnet \
--categoryfile dataset/modelnet40_half1.txt \
--writer False



python -W ignore train.py \
--dataset_type shapenet.sim.easy \
--object all \
--writer False