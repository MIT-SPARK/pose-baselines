#cd ../

python -W ignore test.py \
--outfile logs/shapenet_depth/chair/_test.txt \
--dataset_type shapenet_depth \
--object chair \
--data_type synthetic \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
--writer True