## Note

Training script
```bash
python main.py \
exp_num='0.000' \
training="complete_pcloud" \
dataset="modelnet40_complete" \
category='airplane' \
use_wandb=False \
nr_epochs=10 \
save_frequency=1
```

Test script
```bash 
python main.py \
exp_num='1.002' \
training="complete_pcloud" \
dataset="modelnet40_complete" \
category='airplane' \
eval=True save=True
```

```bash
python main.py \
exp_num='0.001' \
training="complete_pcloud" \
dataset="shapenet_depth" \
category='airplane' \
use_wandb=False \
nr_epochs=10 \
save_frequency=1
```


## Test for all pre-trained objects

Airplane
```bash
python main.py \
exp_num='0.913r' \
training="partial_pcloud" \
dataset="modelnet40_partial" \
category='airplane' \
eval=True \
save=True
```

Car
```bash
python main.py \
exp_num='0.921r' \
training="partial_pcloud" \
dataset="modelnet40_partial" \
category='car' \
eval=True \
save=True
```

Chair 
```bash
python main.py \
exp_num='0.941r' \
training="partial_pcloud" \
dataset="modelnet40_partial" \
category='chair' \
eval=True \
save=True
```

Sofa 
```bash
python main.py \
exp_num='0.951r' \
training="partial_pcloud" \
dataset="modelnet40_partial" \
category='sofa' \
eval=True \
save=True
```

Bottle
```bash
python main.py \
exp_num='0.961r' \
training="partial_pcloud" \
dataset="modelnet40_partial" \
category='bottle' \
eval=True \
save=True
```

