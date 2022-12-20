MODEL_FILE="models/modelnet_noisy.pth"
#MODEL_FILE="runs/Nov29_23-15-54_spark-agent/_model_best.pth"
OBJECTS_SHAPENET='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python test.py --type shapenet.sim.easy \
  --checkpoint $MODEL_FILE \
  --use_rri \
  --object $object_ \
  --final True

done


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python test.py --type shapenet.sim.hard \
  --checkpoint $MODEL_FILE \
  --use_rri \
  --object $object_ \
  --final True

done


#for object_ in $OBJECTS_SHAPENET
#do
#  echo $object_
#
#  python test.py --type shapenet.real.easy \
#  --checkpoint $MODEL_FILE \
#  --use_rri \
#  --object $object_ \
#  --final True
#
#done

for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python test.py --type shapenet.real.hard \
  --checkpoint $MODEL_FILE \
  --use_rri \
  --object $object_ \
  --final True

done


OBJECTS_YCB="001_chips_can 002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can \
             006_mustard_bottle 007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can \
             011_banana 019_pitcher_base 021_bleach_cleanser 035_power_drill 036_wood_block 037_scissors \
             040_large_marker 051_large_clamp 052_extra_large_clamp 061_foam_brick"

for object_ in $OBJECTS_YCB
do
  echo $object_

  python test.py --type ycb.real \
  --checkpoint $MODEL_FILE \
  --use_rri \
  --object $object_ \
  --final True

done


for object_ in $OBJECTS_YCB
do
  echo $object_

  python test.py --type ycb.sim \
  --checkpoint $MODEL_FILE \
  --use_rri \
  --object $object_ \
  --final True

done



















