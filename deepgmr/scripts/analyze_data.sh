python train.py --data_file data/train/modelnet20.h5 --analyze_data True
python train.py --data_file data/train/modelnet40.h5 --analyze_data True

python test.py --data_file data/test/modelnet_noisy.h5 --analyze_data True
python test.py --data_file data/test/modelnet_noisy.h5 --analyze_data True

python train.py --type shapenet.sim.easy --analyze_data True
python train.py --type shapenet.sim.medium --analyze_data True
python train.py --type shapenet.sim.hard --analyze_data True
python train.py --type shapenet.real.easy --analyze_data True
python train.py --type shapenet.real.medium --analyze_data True
python train.py --type shapenet.real.hard --analyze_data True

python test.py --type shapenet.sim.easy --analyze_data True
python test.py --type shapenet.sim.medium --analyze_data True
python test.py --type shapenet.sim.hard --analyze_data True
python test.py --type shapenet.real.easy --analyze_data True
python test.py --type shapenet.real.medium --analyze_data True
python test.py --type shapenet.real.hard --analyze_data True

