MODEL_NAME='./logs/model_trained_on_ModelNet40_model_best.pth'

#python -W ignore test.py \
#--dataset_type shapenet.sim.easy \
#--object all \
#--pretrained $MODEL_NAME \
#--writer True
#
#python -W ignore test.py \
#--dataset_type shapenet.sim.medium \
#--object all \
#--pretrained $MODEL_NAME \
#--writer True
#
#python -W ignore test.py \
#--dataset_type shapenet.sim.hard \
#--object all \
#--pretrained $MODEL_NAME \
#--writer True

python -W ignore test.py \
--dataset_type shapenet.real.easy \
--object all \
--pretrained $MODEL_NAME \
--writer True

python -W ignore test.py \
--dataset_type shapenet.real.medium \
--object all \
--pretrained $MODEL_NAME \
--writer True

python -W ignore test.py \
--dataset_type shapenet.real.hard \
--object all \
--pretrained $MODEL_NAME \
--writer True

MODEL_NAME='./runs/Nov28_05-53-37_spark-agent/_model_best.pth'

python -W ignore test.py \
--dataset_type shapenet.real.hard \
--object all \
--pretrained './runs/Nov28_05-53-37_spark-agent/_model_best.pth' \
--writer True

