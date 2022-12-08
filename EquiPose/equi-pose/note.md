# Note

## Installation

```bash
conda create --name equi-pose python=3.8
source activate equi-pose
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r new_requirements.txt

bash build.sh 

pip install seaborn
pip install tensorboard
pip install notebook
```  

Running build.sh compiles extra CUDA libraries.



## Options in train and test scripts


training: (I think this is pre-trained model)
- complete_pcloud
- partial_pcloud

dataset: (All relevant datasets)
- modelnet40_complete
- modelnet40_partial
- shapenet.sim.easy
- shapenet.sim.medium
- shapenet.sim.hard
- shapenet.real.easy
- shapenet.real.medium
- shapenet.real.hard

category:
- dataset category, as well as pre-trained model category.




## Training Script
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
exp_num='0.813' \
training="complete_pcloud" \
dataset="modelnet40_complete" \
category='airplane' \
eval=True save=True
```

```bash
python main.py \
exp_num='0.913r' \
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

