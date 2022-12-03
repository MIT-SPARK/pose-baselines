#cd ../

python -W ignore test.py \
--outfile logs/shapenet_full/all/_test.txt \
--dataset_type shapenet_full_easy \
--object chair \
--data_type synthetic \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
--writer True


