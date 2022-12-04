
# analyze modelnet train data
#python -W ignore train.py \
#--dataset_path dataset/ModelNet/ \
#--dataset_type modelnet \
#--categoryfile dataset/modelnet40_half1.txt \
#--analyze_data True


# analyze modelnet test data
#python -W ignore test.py \
#--dataset_path dataset/ModelNet/ \
#--categoryfile ./dataset/modelnet40_half2.txt \
#--dataset_type modelnet \
#--analyze_data True


python -W ignore test.py \
--dataset_type shapenet.sim.easy \
--object all \
--writer True \
--analyze_data True

python -W ignore test.py \
--dataset_type shapenet.sim.medium \
--object all \
--writer True \
--analyze_data True

python -W ignore test.py \
--dataset_type shapenet.sim.hard \
--object all \
--writer True \
--analyze_data True

python -W ignore test.py \
--dataset_type shapenet.real.easy \
--object all \
--writer True \
--analyze_data True

python -W ignore test.py \
--dataset_type shapenet.real.medium \
--object all \
--writer True \
--analyze_data True

python -W ignore test.py \
--dataset_type shapenet.real.hard \
--object all \
--writer True \
--analyze_data True