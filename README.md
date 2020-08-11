# Meshing-Point-Clouds-with-IER
[[paper](https://arxiv.org/pdf/2007.09267.pdf)]

Codes for Meshing Point Clouds with Predicted Intrinsic-Extrinsic Ratio Guidance (ECCV2020).

![](/teaser.jpg)

We propose a novel mesh reconstruction method that leverages the input point cloud as much as possible, by predicting which triplets of points should form faces. Our key innovation is a surrogate of local connectivity, calculated by comparing the intrinsic/extrinsic metrics. We learn to classify the candidate triangles using a deep network and then feed the results to a post-processing module for mesh generation. Our method can not only preserve fine-grained details, handle ambiguous structures, but also possess strong generalizability to unseen categories.


Codes will be available soon...



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
