python main.py \
eval=True \
new_dataset='shapenet.sim.easy.airplane' \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True

python main.py \
eval=True \
new_dataset='shapenet.sim.medium.airplane' \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True

python main.py \
eval=True \
new_dataset='shapenet.sim.hard.airplane' \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True


##
python main.py \
eval=True \
new_dataset='shapenet.real.easy.airplane' \
dataset='modelnet40_partial' \
category='airplane' \
exp_num='0.913r' \
training='partial_pcloud' \
writer=True

python main.py \
eval=True \
new_dataset='shapenet.real.medium.airplane' \
dataset='modelnet40_partial' \
category='airplane' \
exp_num='0.913r' \
training='partial_pcloud' \
writer=True


python main.py \
eval=True \
new_dataset='shapenet.real.hard.airplane' \
dataset='modelnet40_partial' \
category='airplane' \
exp_num='0.913r' \
training='partial_pcloud' \
writer=True