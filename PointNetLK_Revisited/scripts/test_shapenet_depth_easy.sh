#cd ../

#python -W ignore test.py \
#--outfile logs/shapenet_depth/all/_test.txt \
#--dataset_type shapenet_depth_easy \
#--object all \
#--data_type synthetic \
#--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
#--writer True


python -W ignore test.py \
--dataset_type shapenet_depth_easy \
--object all \
--pretrained './runs/Nov28_05-53-37_spark-agent/_model_best.pth' \
--writer True