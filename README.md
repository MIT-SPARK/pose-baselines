# Point Cloud-Only Pose Estimation
Author: Rajat Talak

We evaluate various methods of estimating object pose, given partial object point cloud and 
the object CAD model. 


## Visualizing the Results 

Install conda environment:
```bash
conda create --name viz python=3.9
conda install pip 
pip install numpy
pip install pandas
pip install seaborn
pip install notebook
pip install ipywidgets
```

Make sure you are in the room directory (i.e. pose-baselines/). Run jupyter notebook:
```bash
jupyter notebook
```
Open results.ipynb and run. You should see the following:






## Evaluating Models

### DeepGMR

Install conda environment:
```bash
cd deepgmr
conda env create -f environment.yml 
conda activate deepgmr
conda install -c open3d-admin -c conda-forge open3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
````

Run evaluations:
```bash
cd deepgmr
conda activate deepgmr
python scripts/test.sh
```


### PointNetLK (Revisited)

Install conda environment:
```bash
cd PointNetLK_Revisited
conda create --name lk python=3.8
conda activate lk
conda install pip git
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
```

Run evaluations:
```bash
cd PointNetLK_Revisited
conda activate lk
python scripts/test.sh
```


### EquiPose

Install conda environment:
```bash
cd EquiPose/equi-pose/
conda create --name equi-pose python=3.8
source activate equi-pose
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r new_requirements.txt
bash build.sh 
pip install seaborn
pip install tensorboard
```

Run evaluations:
```bash
cd EquiPose/equi-pose/
conda activate equi-pose 
python scripts/test.sh
```


### FPFH + TEASER++

Install conda environment:
```bash
conda create --name fpfh python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
conda install pip
pip install open3d
```

Install TEASER++ by following the instructions in [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus). 
Clone the TEASER++ repo in a new directory:
```bash
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git 
```

Install TEASER++ in fpfh conda environment as follows:
```bash
sudo apt install cmake libeigen3-dev libboost-all-dev

conda activate fpfh 
cd TEASER-plusplus

cd TEASER-plusplus && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=<version> .. && make teaserpp_python
cd python && pip install .
cd ../.. && cd examples/teaser_python_ply 
python teaser_python_ply.py
```
Make sure to replace \<version\> with the python version in fpfh.

Run evaluations:
```bash
cd fpfh_teaser
conda activate fpfh
python scripts/test.sh
```

