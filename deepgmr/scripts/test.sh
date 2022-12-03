#MODEL_FILE="models/modelnet_noisy.pth"
MODEL_FILE="runs/Nov29_23-15-54_spark-agent/_model_best.pth"

python test.py --data_file data/test/modelnet_noisy.h5 \
--checkpoint $MODEL_FILE \
--use_rri

python test.py --type shapenet.sim.easy \
--checkpoint $MODEL_FILE \
--use_rri \
--object all

python test.py --type shapenet.sim.medium \
--checkpoint $MODEL_FILE \
--use_rri \
--object all

python test.py --type shapenet.sim.hard \
--checkpoint $MODEL_FILE \
--use_rri \
--object all

python test.py --type shapenet.real.easy \
--checkpoint $MODEL_FILE \
--use_rri \
--object all

python test.py --type shapenet.real.medium \
--checkpoint $MODEL_FILE \
--use_rri \
--object all

python test.py --type shapenet.real.hard \
--checkpoint $MODEL_FILE \
--use_rri \
--object all