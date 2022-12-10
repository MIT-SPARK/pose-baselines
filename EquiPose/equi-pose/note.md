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


