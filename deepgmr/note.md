# Running ShapeNet baseline

Install environment with open3d.
``bash
conda env create -f environment.yml 
conda activate <env-name>
conda install -c open3d-admin -c conda-forge open3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
``

Run training on sim and real dataset:
```bash
python train.py --type shapenet.real --log_dir log/shapenet_real/train --use_rri
python train.py --type shapenet.sim --log_dir log/shapenet_sim/train --use_rri
```

If you need to train and already trained model further for more epochs, use:
```bash
python train.py --type shapenet.real \
  --log_dir log/shapenet_real/train \
  --use_rri \
  --pre_trained_model log/shapenet_real/train/checkpoint_epoch-100.pth \
  --n_epochs 200
```



Test models on shapenet.real dataset:
```bash
python test.py --type shapenet.real --checkpoint models/shapenet_real.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/shapenet_real/test
```

Test pre-trained models on the shapenet.real dataset:
```bash
python test.py --type shapenet.real --checkpoint models/modelnet_clean.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/test/modelnet_clean
```

```bash
python test.py --type shapenet.real --checkpoint models/modelnet_noisy.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/test/modelnet_noisy
```

```bash
python test.py --type shapenet.real --checkpoint models/modelnet_unseen.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/test/modelnet_unseen
```




Trying ---

Sim trained model evaluating on sim dataset
```bash
python test.py --type shapenet.sim --checkpoint models/shapenet_sim-checkpoint_epoch-100.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/shapenet_sim/test \
--object 'chair'
```

Sim trained model evaluating on real dataset
```bash
python test.py --type shapenet.real --checkpoint models/shapenet_sim-checkpoint_epoch-100.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/shapenet_real/test \
--object 'chair'
```

Real trained model evaluating on real dataset
```bash
python test.py --type shapenet.real --checkpoint models/shapenet_real-checkpoint_epoch-150.pth \
--use_rri \
--save_results \
--results_dir log/shapenet_real/shapenet_real/test \
--object 'chair'
```























