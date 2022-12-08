DATA='modelnet40_complete'
MODELS=('/model/modelnet40_complete/0.813/' \
'/model/modelnet40_complete/0.851/' \
'/model/modelnet40_complete/0.8562/' \
'/model/modelnet40_complete/0.8581/' \
'/model/modelnet40_complete/0.8591/' \
'/model/modelnet40_partial/0.913r/' \
'/model/modelnet40_partial/0.921r/'  \
'/model/modelnet40_partial/0.941r/' \
'/model/modelnet40_partial/0.951r/' \
'/model/modelnet40_partial/0.961r/' \
)



python main.py \
eval=True \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True \
analyze_data=True


python main.py \
eval=True \
dataset='modelnet40_complete' \
category='car' \
exp_num='0.851' \
training='complete_pcloud' \
writer=True


python main.py \
eval=True \
dataset='modelnet40_complete' \
category='bottle' \
exp_num='0.8562' \
training='complete_pcloud' \
writer=True


python main.py \
eval=True \
dataset='modelnet40_complete' \
category='chair' \
exp_num='0.8581' \
training='complete_pcloud' \
writer=True


python main.py \
eval=True \
dataset='modelnet40_complete' \
category='sofa' \
exp_num='0.8591' \
training='complete_pcloud' \
writer=True


##
python main.py \
eval=True \
dataset='modelnet40_partial' \
category='airplane' \
exp_num='0.913r' \
training='partial_pcloud' \
writer=True \
analyze_data=True

python main.py \
eval=True \
dataset='modelnet40_partial' \
category='car' \
exp_num='0.921r' \
training='partial_pcloud' \
writer=True

python main.py \
eval=True \
dataset='modelnet40_partial' \
category='bottle' \
exp_num='0.941r' \
training='partial_pcloud' \
writer=True

python main.py \
eval=True \
dataset='modelnet40_partial' \
category='chair' \
exp_num='0.951r' \
training='partial_pcloud' \
writer=True

python main.py \
eval=True \
dataset='modelnet40_partial' \
category='sofa' \
exp_num='0.961r' \
training='partial_pcloud' \
writer=True


# note
# - their code links dataset to model. we cannot evaluate their model, on any other dataset.
# - example: a complete_pcloud trained model, cannot be easily evaluated on modelnet40_partial
# - I have added a fix, where I define new_dataset, and I test on my dataset.






#python main.py \
#eval=True \
#dataset='modelnet40_complete' \
#category='sofa' \
#exp_num='0.8591' \
#training='complete_pcloud' \
#writer=True \
#new_dataset='shapenet.sim.easy.chair'


#python main.py \
#eval=True \
#dataset='modelnet40_complete' \
#category='sofa' \
#exp_num='0.8591' \
#training='complete_pcloud' \
#writer=True \
#new_dataset='ycb.real.006_mustard_bottle'














