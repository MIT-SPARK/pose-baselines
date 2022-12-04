
Installing environment
```bash
conda create --namel lk python=3.8
conda activate lk
conda install pip git
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
```

My installations: seaborn, jupyter
```bash
pip install seaborn 
pip install notebook
pip install tensorboard 
```
- tensorboard
- seaborn
- jupyter notebook


Pytorch3D, I couldn't install
```bash 
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub 
conda install pytorch3d -c pytorch3d 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
```


Testing pre-trained model on their modelnet dataset
```bash
python test.py \
--outfile "./test_logs/temp_test" \
--dataset_path "./dataset/ModelNet" \
--categoryfile "./dataset/modelnet40_half2.txt" \
--pose_file "./dataset/gt_poses.csv" \
--dataset_type "modelnet" \
--data_type "real" \
--num_points 10000 \
--batch_size 1 \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth'
```

Testing pre-trained model on our shapenet dataset
```bash
python test.py \
--outfile "./test_logs/temp_test" \
--dataset_type "shapenet" \
--object "chair" \
--data_type "real" \
--num_points 1000 \
--batch_size 1 \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth'
```

Training further, a pre-trained model on our shapenet dataset
```bash
python train.py \
--outfile "./test_logs/temp_test" \
--dataset_path "./dataset/ModelNet" \
--categoryfile "./dataset/modelnet40_half2.txt" \
--dataset_type "modelnet" \
--data_type "real" \
--num_points 1000 \
--batch_size 1 \
--pretrained './logs/model_trained_on_ModelNet40_model_best.pth'
#--pose_file "./dataset/gt_poses.csv" \
```

### ToDo:
Verify the working of the above procedure. Replicate paper results.
Test the above running.
Setup tensorboard.
Test our dataset.
Testing our trained model.
Setup training for each object.
