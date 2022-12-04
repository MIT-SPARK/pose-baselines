# cd ../

python -W ignore test.py \
--dataset_path dataset/ModelNet/ \
--categoryfile ./dataset/modelnet40_half2.txt \
--dataset_type modelnet \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
--writer False




#python -W ignore test.py \
#--outfile logs/modelnet/_test.txt \
#--dataset_path dataset/ModelNet/ \
#--categoryfile ./dataset/modelnet40_half2.txt \
#--dataset_type modelnet \
#--data_type synthetic \
#--pretrained './runs/Nov27_00-43-16_spark-agent/_model_best.pth' \
#--writer True