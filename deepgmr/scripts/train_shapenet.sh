python train.py --type shapenet.real.medium \
--pre_trained_model models/modelnet_noisy.pth \
--use_rri  \
--n_epochs 300



python train.py --type shapenet.real.hard \
--pre_trained_model models/modelnet_noisy.pth \
--use_rri  \
--n_epochs 300




python train.py --type shapenet.sim.medium \
--pre_trained_model models/modelnet_noisy.pth \
--use_rri  \
--n_epochs 300