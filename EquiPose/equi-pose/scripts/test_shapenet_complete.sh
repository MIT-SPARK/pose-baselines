
## airplane
python main.py \
eval=True \
new_dataset='shapenet.sim.easy.airplane' \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True \
final=True

python main.py \
eval=True \
new_dataset='shapenet.sim.hard.airplane' \
dataset='modelnet40_complete' \
category='airplane' \
exp_num='0.813' \
training='complete_pcloud' \
writer=True \
final=True


## car
python main.py \
eval=True \
new_dataset='shapenet.sim.easy.car' \
dataset='modelnet40_complete' \
category='car' \
exp_num='0.851' \
training='complete_pcloud' \
writer=True \
final=True

python main.py \
eval=True \
new_dataset='shapenet.sim.hard.car' \
dataset='modelnet40_complete' \
category='car' \
exp_num='0.851' \
training='complete_pcloud' \
writer=True \
final=True


## bottle
python main.py \
eval=True \
new_dataset='shapenet.sim.easy.bottle' \
dataset='modelnet40_complete' \
category='bottle' \
exp_num='0.8562' \
training='complete_pcloud' \
writer=True \
final=True

python main.py \
eval=True \
new_dataset='shapenet.sim.hard.bottle' \
dataset='modelnet40_complete' \
category='bottle' \
exp_num='0.8562' \
training='complete_pcloud' \
writer=True \
final=True


## chair
python main.py \
eval=True \
new_dataset='shapenet.sim.easy.chair' \
dataset='modelnet40_complete' \
category='chair' \
exp_num='0.8581' \
training='complete_pcloud' \
writer=True \
final=True

python main.py \
eval=True \
new_dataset='shapenet.sim.hard.chair' \
dataset='modelnet40_complete' \
category='chair' \
exp_num='0.8581' \
training='complete_pcloud' \
writer=True \
final=True