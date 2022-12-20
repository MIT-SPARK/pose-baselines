OBJECTS_SHAPENET='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python evaluate_fpfh_teaser_icp.py \
  --dataset shapenet.sim.easy \
  --object $object_ \
  --final True

done


for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python evaluate_fpfh_teaser_icp.py \
  --dataset shapenet.sim.hard \
  --object $object_ \
  --final True

done


#for object_ in $OBJECTS_SHAPENET
#do
#  echo $object_
#
#  python evaluate_fpfh_teaser_icp.py \
#  --dataset shapenet.real.easy \
#  --object $object_ \
#  --final True
#
#done

for object_ in $OBJECTS_SHAPENET
do
  echo $object_

  python evaluate_fpfh_teaser_icp.py \
  --dataset shapenet.real.hard \
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

  python evaluate_fpfh_teaser_icp.py \
  --dataset ycb.real \
  --object $object_ \
  --final True

done


for object_ in $OBJECTS_YCB
do
  echo $object_

  python evaluate_fpfh_teaser_icp.py \
  --dataset ycb.sim \
  --object $object_ \
  --final True

done



















