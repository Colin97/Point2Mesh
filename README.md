# Meshing-Point-Clouds-with-IER
Codes for Meshing Point Clouds with Predicted Intrinsic-Extrinsic Ratio Guidance (ECCV2020).
[[paper](https://arxiv.org/pdf/2007.09267.pdf)]

![](/teaser.jpg)

We propose a novel mesh reconstruction method that leverages the input point cloud as much as possible, by predicting which triplets of points should form faces. Our key innovation is a surrogate of local connectivity, calculated by comparing the intrinsic/extrinsic metrics. We learn to classify the candidate triangles using a deep network and then feed the results to a post-processing module for mesh generation. Our method can not only preserve fine-grained details, handle ambiguous structures, but also possess strong generalizability to unseen categories.

### 0. Envrionment & Prerequisites.
a) Environment:
	- PyTorch 1.3.1
	- Python 3.6
	- Cuda 10.0

b) Download submodules [annoy(1.16)](https://github.com/spotify/annoy/) and [SparseConvNet(0.2)](https://github.com/facebookresearch/SparseConvNet/) and install SparseConvNet:

```
git submodule update --init --recursive
cd SparseConvNet/
sh develop.sh
```
**annoy 1.17 changed their API. Please download the [previous version](https://github.com/spotify/annoy/archive/v1.16.zip).**

c) Install [plyfile](https://github.com/dranjan/python-plyfile), pickle, and tqdm with pip.

### 1. Download pretrained models and demo data.
You can download the pretrained model and demo data from [here](https://drive.google.com/drive/folders/1Wb-mU3mcxpKAQyb7LqqQYKbfnpmDXSiZ?usp=sharing) to get a quick look. Demo data includes ten shapes (both gt mesh and point cloud) and their pre-generated pickle files. The pickle files contain the point cloud vertices and proposed candidate triangles (vertex indices and gt labels). You can use the pickles files to train or test the network.

### 2. Classify proposed candidate triangles with a neural network.
You can use `network/test.py` to classify the proposed candidate triangles. You can find the prediced labels (npy files) at `log/shapenet_pretrained/test_demo`. 300,000 triangles per npy file and each shape may have multiple npy files.

### 3. Post-process and get output meshes.

You can feed the pickle files and the predicted npy files into a post-process program to get output meshes.

First, compile cpp codes:

```
cd postprocess
mkdir build
cd build
cmake ..
make
cd ..
```
Then, you can post-process all the demo shapes with `run_demo.py` or post-process a single shape with `main.py`. You can find the generated demo meshes at `log/shapenet_pretrained/test_demo/output_mesh`.

### 4. Train your own network.
You can download all the pickle files for the full ShapeNet dataset from [here](https://drive.google.com/drive/folders/1Wb-mU3mcxpKAQyb7LqqQYKbfnpmDXSiZ?usp=sharing)(23,108 shapes, ~42.2GB). Then use `network/train.py` to train your own network.

### 5. Generate your own training data.

You can generate your own training data with gt mesh (ply). 

First, compile the cpp code:

```
cd preprocess_with_gt_mesh
mkdir build
cd build
cmake ..
make
cd ..
```
Then, you can use `main.py` to generate the picke file for a single shape or use `run_demo.py` to generate the pickle files for all the demo meshes. The total runtime for each shape may take several minutes. You can use multiple processes to accelerate. 

In detail, the training data generation consists of several steps: 
- preprocess the gt mesh: normalize mesh, merge close vertices, etc. 
- sample point cloud: sample 12,000 ~ 12,800 points with Poisson sampling and use binary search to determine the radius.
- calculate geodesic distance between pairs of points: it may take up to 1 minute. In some cases (e.g., complex and broken meshes), it may time out and thus fail to generate the final pickle file.
- propose candidate triangles based on KNN.
- calculate the distances between the candidate triangles and ground truth mesh.

### 6. Generate pickle files with only point clouds.
You can also generate pickle files with only point clouds (ply), so that you can feed the pickle files into the network and the postprocess program to get the final mesh. 

First, compile the cpp code:
```
cd preprocess_with_pc
mkdir build
cd build
cmake ..
make
cd ..
```

Then, you can use `main.py` to generate the picke file for a single shape or use `run_demo.py` to generate the pickle files for all the demo point clouds. The total runtime for each shape may take less than one minute. You can use multiple processes to accelerate. Please note that, in this way, the candidate labels in the pickle files will be set to -1. 

The input point cloud should contain 12,000 ~ 12,800 points (to best fit our pre-trained network). Using Poisson sampling as pre-processing can get evenly distributed point cloud and thus boost the performance. Currently, our method do not support very noisy point clouds.

If you find our work useful for your research, please cite:

```
@article{liu2020meshing,
  title={Meshing Point Clouds with Predicted Intrinsic-Extrinsic Ratio Guidance},
  author={Liu, Minghua and Zhang, Xiaoshuai and Su, Hao},
  journal={arXiv preprint arXiv:2007.09267},
  year={2020}
}
```

### 
