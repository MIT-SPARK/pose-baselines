MODEL_NAME='./logs/model_trained_on_ModelNet40_model_best.pth'
OBJECTS_SHAPENET='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python -W ignore test.py \
  --dataset_type shapenet.sim.easy \
  --object $object_ \
  --pretrained $MODEL_NAME \
  --writer True \
  --final True

done


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python -W ignore test.py \
  --dataset_type shapenet.sim.hard \
  --object $object_ \
  --pretrained $MODEL_NAME \
  --writer True \
  --final True

done


#for object_ in $OBJECTS_SHAPENET
#do
#  echo $object_
#
#  python -W ignore test.py \
#  --dataset_type shapenet.real.easy \
#  --object $object_ \
#  --pretrained $MODEL_NAME \
#  --writer True \
#  --final True
#
#done

for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python -W ignore test.py \
  --dataset_type shapenet.real.hard \
  --object $object_ \
  --pretrained $MODEL_NAME \
  --writer True \
  --final True

done


OBJECTS_YCB="001_chips_can 002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can \
             006_mustard_bottle 007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can \
             011_banana 019_pitcher_base 021_bleach_cleanser 035_power_drill 036_wood_block 037_scissors \
             040_large_marker 051_large_clamp 052_extra_large_clamp 061_foam_brick"

for object_ in $OBJECTS_YCB
do
  echo $object_

  python -W ignore test.py \
  --dataset_type ycb.real \
  --object $object_ \
  --pretrained $MODEL_NAME \
  --writer False \
  --final True

done


for object_ in $OBJECTS_YCB
do
  echo $object_

  python -W ignore test.py \
  --dataset_type ycb.sim \
  --object $object_ \
  --pretrained $MODEL_NAME \
  --writer False \
  --final True

done



















