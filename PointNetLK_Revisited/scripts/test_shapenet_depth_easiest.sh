#cd ../

python -W ignore test.py \
--outfile logs/shapenet_depth/all/_test.txt \
--dataset_type shapenet_depth_easiest \
--object all \
--data_type synthetic \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
--writer True
