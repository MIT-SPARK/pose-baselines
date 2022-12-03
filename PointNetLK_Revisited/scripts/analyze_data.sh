
# analyze modelnet train data
python -W ignore train.py \
--outfile logs/modelnet/ \
--dataset_path dataset/ModelNet/ \
--dataset_type modelnet \
--data_type synthetic \
--categoryfile dataset/modelnet40_half1.txt \
--analyze_data True


# analyze modelnet test data
python -W ignore test.py \
--outfile logs/modelnet/_test.txt \
--dataset_path dataset/ModelNet/ \
--categoryfile ./dataset/modelnet40_half2.txt \
--dataset_type modelnet \
--data_type synthetic \
--analyze_data True


# analyze shapenet train/test data
python -W ignore train.py \
--outfile logs/shapenet_full/all/ \
--dataset_type shapenet_full \
--object all \
--data_type synthetic \
--analyze_data True



# analyze shapenet easy train/test data
python -W ignore train.py \
--outfile logs/shapenet_full_easy/all/ \
--dataset_type shapenet_full_easy \
--object all \
--data_type synthetic \
--analyze_data True
