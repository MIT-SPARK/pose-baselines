# note: this code trains the fcgf and then evaluates for different voxel sizes

cd ../../c3po/baselines

#python fcgf_train_shapenet.py \
#--type 'real' \
#--voxel_size 0.01 \
#--out_dir "fcgf/data/output_shapenet/realv0010" \
#--model_n_out 16 \
#--conv1_kernel_size 3 \
#--train_data_len_shapenet 2048 \
#--val_data_len_shapenet 512 \
#--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"
#
#python fcgf_train_shapenet.py \
#--type 'real' \
#--voxel_size 0.025 \
#--out_dir "fcgf/data/output_shapenet/realv0025" \
#--model_n_out 16 \
#--conv1_kernel_size 3 \
#--train_data_len_shapenet 2048 \
#--val_data_len_shapenet 512 \
#--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"
#
#python fcgf_train_shapenet.py \
#--type 'real' \
#--voxel_size 0.05 \
#--out_dir "fcgf/data/output_shapenet/realv0050" \
#--model_n_out 16 \
#--conv1_kernel_size 3 \
#--train_data_len_shapenet 2048 \
#--val_data_len_shapenet 512 \
#--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"

#python fcgf_train_shapenet.py \
#--type 'real' \
#--voxel_size 0.075 \
#--out_dir "fcgf/data/output_shapenet/realv0075" \
#--model_n_out 16 \
#--conv1_kernel_size 3 \
#--train_data_len_shapenet 2048 \
#--val_data_len_shapenet 512 \
#--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"
#
#python fcgf_train_shapenet.py \
#--type 'real' \
#--voxel_size 0.100 \
#--out_dir "fcgf/data/output_shapenet/realv0100" \
#--model_n_out 16 \
#--conv1_kernel_size 3 \
#--train_data_len_shapenet 2048 \
#--val_data_len_shapenet 512 \
#--weights "fcgf/ResUNetBN2C-16feat-3conv.pth"

cd ../expt_shapenet/


##
#OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
#FILE_NAME="./eval/eval_teaser_fcgf_icp_v0010.log"
#MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0010/real/best_val_checkpoint.pth"
#MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0010/real"
#
#
## Computes ADD-S, ADD-S AUC, and % certifiable for each object category
#FOLDER_NAME="./temp"
#mkdir $FOLDER_NAME
#
#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#for OBJECT in $OBJECT_LIST
#do
#  echo $OBJECT
#  python -u evaluate_teaser_fcgf_icp.py \
#  --object $OBJECT \
#  --model $MODEL_FILE \
#  --pre y \
#  --folder $FOLDER_NAME >> $FILE_NAME
#
#  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
#  # The default is 'y'.
#done
#
#python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0010"
#
#rm -r $FOLDER_NAME


##
#OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
#FILE_NAME="./eval/eval_teaser_fcgf_icp_v0025.log"
#MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0025/real/best_val_checkpoint.pth"
#MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0025/real"
#
#
## Computes ADD-S, ADD-S AUC, and % certifiable for each object category
#FOLDER_NAME="./temp"
#mkdir $FOLDER_NAME
#
#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#for OBJECT in $OBJECT_LIST
#do
#  echo $OBJECT
#  python -u evaluate_teaser_fcgf_icp.py \
#  --object $OBJECT \
#  --model $MODEL_FILE \
#  --pre y \
#  --folder $FOLDER_NAME >> $FILE_NAME
#
#  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
#  # The default is 'y'.
#done
#
#python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0025"
#
#rm -r $FOLDER_NAME

##
#OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
#FILE_NAME="./eval/eval_teaser_fcgf_icp_v0050.log"
#MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0050/real/best_val_checkpoint.pth"
#MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0050/real"
#
#
## Computes ADD-S, ADD-S AUC, and % certifiable for each object category
#FOLDER_NAME="./temp"
#mkdir $FOLDER_NAME
#
#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#for OBJECT in $OBJECT_LIST
#do
#  echo $OBJECT
#  python -u evaluate_teaser_fcgf_icp.py \
#  --object $OBJECT \
#  --model $MODEL_FILE \
#  --pre y \
#  --folder $FOLDER_NAME >> $FILE_NAME
#
#  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
#  # The default is 'y'.
#done
#
#python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0050"
#
#rm -r $FOLDER_NAME


###
#OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
#FILE_NAME="./eval/eval_teaser_fcgf_icp_v0075.log"
#MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0075/real/best_val_checkpoint.pth"
#MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0075/real"
#
#
## Computes ADD-S, ADD-S AUC, and % certifiable for each object category
#FOLDER_NAME="./temp"
#mkdir $FOLDER_NAME
#
#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#for OBJECT in $OBJECT_LIST
#do
#  echo $OBJECT
#  python -u evaluate_teaser_fcgf_icp.py \
#  --object $OBJECT \
#  --model $MODEL_FILE \
#  --pre y \
#  --folder $FOLDER_NAME >> $FILE_NAME
#
#  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
#  # The default is 'y'.
#done
#
#python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0075"
#
#rm -r $FOLDER_NAME


###
#OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
#FILE_NAME="./eval/eval_teaser_fcgf_icp_v0100.log"
#MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0100/real/best_val_checkpoint.pth"
#MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0100/real"
#
#
## Computes ADD-S, ADD-S AUC, and % certifiable for each object category
#FOLDER_NAME="./temp"
#mkdir $FOLDER_NAME
#
#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#for OBJECT in $OBJECT_LIST
#do
#  echo $OBJECT
#  python -u evaluate_teaser_fcgf_icp.py \
#  --object $OBJECT \
#  --model $MODEL_FILE \
#  --pre y \
#  --folder $FOLDER_NAME >> $FILE_NAME
#
#  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
#  # The default is 'y'.
#done
#
#python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0100"
#
#rm -r $FOLDER_NAME


OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
FILE_NAME="./eval/eval_teaser_fcgf_icp_v0050_noicp.log"
MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv0050/real/best_val_checkpoint.pth"
MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv0050/real"


# Computes ADD-S, ADD-S AUC, and % certifiable for each object category
FOLDER_NAME="./temp"
mkdir $FOLDER_NAME

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "TEASER + FCGF: baseline" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

for OBJECT in $OBJECT_LIST
do
  echo $OBJECT
  python -u evaluate_teaser_fcgf_icp.py \
  --object $OBJECT \
  --model $MODEL_FILE \
  --pre y \
  --folder $FOLDER_NAME >> $FILE_NAME

  # note: had added icp=False in teaser_fcgf_icp, for this evaluation
done

python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "resultsv0050_noicp"

rm -r $FOLDER_NAME