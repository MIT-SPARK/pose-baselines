# cd ../

python -W ignore test.py \
--outfile logs/modelnet/_test.txt \
--dataset_path dataset/ModelNet/ \
--categoryfile ./dataset/modelnet40_half1.txt \
--dataset_type modelnet_train \
--data_type synthetic \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth' \
--writer True


#python -W ignore test.py \
#--outfile logs/modelnet/_test.txt \
#--dataset_path dataset/ModelNet/ \
#--categoryfile ./dataset/modelnet40_half1.txt \
#--dataset_type modelnet_train \
#--data_type synthetic \
#--pretrained './runs/Nov27_00-43-16_spark-agent/_model_best.pth' \
#--writer True