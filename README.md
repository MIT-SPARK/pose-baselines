# Point Cloud-Only Pose Estimation
Author: Rajat Talak

We evaluate various methods of estimating object pose, given partial object point cloud and 
the object CAD model. We compare various methods in the literature, namely,
[TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus),
[DeepGMR](https://github.com/wentaoyuan/deepgmr),
[PointNetLK](https://github.com/Lilac-Lee/PointNetLK_Revisited),
[EquiPose](https://github.com/dragonlong/equi-pose),
and several
semantic keypoint-based methods used in [C-3PO](https://github.com/MIT-SPARK/C-3PO). 
We would like to thank the Authors of these repos, as we use their implementation and training setup, to generate results.


## Visualizing the Results 

Clone the repository:
```bash
git clone --depth 1 https://github.com/MIT-SPARK/pose-baselines.git
cd pose-baselines/
```

Install conda environment:
```bash
conda create --name pb-viz python=3.9
conda activate pb-viz
conda install pip 
pip install numpy
pip install pandas
pip install seaborn
pip install notebook
pip install ipywidgets
pip install pickle5
```

Make sure you are in the room directory (i.e. pose-baselines/). Run jupyter notebook:
```bash
conda activate pb-viz
jupyter notebook results.ipynb
```
You should see the following:

![preview](docs/pd-results.gif) 



## Evaluating Models

### Adding Datasets

Create two empty folders named data_shapenet and data_ycb in:
1. pose-baselines/deepgmr/
2. pose-baselines/EquiPose/equi-pose/
3. pose-baselines/fpfh_teaser/
4. pose-baselines/PointNetLK_Revisited/

Our experiments rely on  the [ShapeNet](https://shapenet.org/), [KeypointNet](https://github.com/qq456cvb/KeypointNet), 
and the [YCB](https://www.ycbbenchmarks.com/object-models/) datasets. 
Please view ShapeNet's terms of use [here](https://shapenet.org/terms). 
There's no need to download the datasets seperately. 
Our experiments use a processed version of these datasets. 
Follow the steps below to download and save the relevant dataset. 

Download our processed dataset files on Google Drive [here](https://drive.google.com/drive/folders/1EYa8B0dID1vk9bze93pzil8rVj2-fYb5?usp=sharing). 
We've provided the dataset as a zip archive split into 1GB chunks of the format ```c3po-data.z**```.
Combine and unzip the folder:

```bash
zip -F c3po-data.zip --out data.zip
unzip data.zip
```

Verify your directory structure looks as follows:
```
data
|
|─── learning-objects
│  
│─── KeypointNet
│   
└─── ycb
```
Move the KeypointNet and learning-objects folders in data_shapenet and ycb folder in data_ycb. You may even create symlink.


### FPFH + TEASER++

Install conda environment:
```bash
cd fpfh_teaser
conda env create -f environment.yml
conda activate pb-fpfh
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
cmake -DTEASERPP_PYTHON_VERSION=3.8 .. && make teaserpp_python
cd python && pip install .
```
This installs teaserpp_python.

Run evaluations:
```bash
cd fpfh_teaser
conda activate pb-fpfh
bash scripts/test.sh
```
note: uncomment out "from pytorch3d import ops" in pose-baselines/utils_common.py

### DeepGMR

Install conda environment:
```bash
cd deepgmr
conda env create -f environment.yml 
conda activate pb-deepgmr
conda install -c open3d-admin -c conda-forge open3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pickle5 
pip install seaborn
````

Run evaluations:
```bash
cd deepgmr
conda activate pb-deepgmr
bash scripts/test.sh
```
note: uncomment out "from pytorch3d import ops" in pose-baselines/utils_common.py

### PointNetLK (Revisited)

Install conda environment:
```bash
cd PointNetLK_Revisited
conda create --name pb-lk python=3.8
conda activate pb-lk
conda install pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install tensorboard 
pip install pickle5 
pip install seaborn
```

Run evaluations:
```bash
cd PointNetLK_Revisited
conda activate pb-lk
bash scripts/test.sh
```
note: either comment out "from pytorch3d import ops" in pose-baselines/utils_common.py or install pytorch3d in pb-lk.

### EquiPose

Install conda environment:
```bash
cd EquiPose/equi-pose/
conda create --name pb-equi-pose python=3.8
conda activate pb-equi-pose
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r new_requirements.txt
bash build.sh
pip install tensorboard
```
or install the environment as specified in pose-baselines/EquiPose/equi=pose/README.md. 

Download the pre-trained models [drive_link](https://drive.google.com/drive/folders/1i8EvIugHF8kmk-sgpzAhiQM2a4p1R7cu) 
provided by [EquiPose](https://github.com/dragonlong/equi-pose). Add them to the folder: pose-baselines/EquiPose/logs/model/

Run evaluations:
```bash
cd EquiPose/equi-pose/
conda activate pb-equi-pose 
bash scripts/test.sh
```
note: either comment out "from pytorch3d import ops" in pose-baselines/utils_common.py or install pytorch3d in pb-equi-pose.


## Citation

If you find this project useful, do cite our work:

```bibtex
@article{Talak22arxiv-correctAndCertify,
  title = {Correct and {{Certify}}: {{A New Approach}} to {{Self-Supervised 3D-Object Perception}}},
  author = {Talak, Rajat and Peng, Lisa and Carlone, Luca},
  year = {2022},
  month = {Jun.},
  journal = {arXiv preprint arXiv: 2206.11215},
  eprint = {2206.11215},
  note = {\linkToPdf{https://arxiv.org/pdf/2206.11215.pdf}},
  pdf={https://arxiv.org/pdf/2206.11215.pdf},
  Year = {2022}
}

```


## License
This project is released under MIT license.


## Acknowledgement
This work was partially funded by ARL DCIST CRA W911NF-17-2-0181, ONR RAIDER N00014-18-1-2828, and NSF CAREER award "Certifiable Perception for Autonomous Cyber-Physical Systems".
