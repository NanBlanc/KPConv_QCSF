**Adaptation of KPConv to work with KITTI-360**

**First phase of my M. Sc. degree**

Knowledge of the semantic classes present in the scene aids in the detection and characterization of occlusions. Thus, the first phase of the methodology aims to perform semantic segmentation of the point clouds. In the context of this project, it is considered that these point clouds are related only to the residential area (i.e., from the property boundaries to the road), and the availability of ground footprints of buildings allows extracting these areas from the complete point cloud. Neural networks that have demonstrated good performance with dense point clouds in outdoor environments will be considered for semantic segmentation. There are several challenges related to semantic segmentation. The first challenge involves selecting the appropriate classes for the subsequent detection of occlusions. This will include determining the granularity of these classes (e.g., building versus facade, roof). The second challenge involves training the networks with publicly available datasets (e.g., KITTI-360). It will be necessary to adapt the segmentation approach to obtain satisfactory results for residential and rural areas, as KpConv is not suited for aggregated point clouds.


.........

![Intro figure](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

This repository contains the implementation of **Kernel Point Convolution** (KPConv) in [PyTorch](https://pytorch.org/).

KPConv is also available in [Tensorflow](https://github.com/HuguesTHOMAS/KPConv) (original but older implementation).

Another implementation of KPConv is available in [PyTorch-Points-3D](https://github.com/nicolas-chaulet/torch-points3d)
 
KPConv is a point convolution operator presented in our ICCV2019 paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find our work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

## Installation

This implementation has been tested on Ubuntu 18.04 and Windows 10. Details are provided in [INSTALL.md](./INSTALL.md).


## Experiments

We provide scripts for three experiments: ModelNet40, S3DIS and SemanticKitti. The instructions to run these 
experiments are in the [doc](./doc) folder.

* [Object Classification](./doc/object_classification_guide.md): Instructions to train KP-CNN on an object classification
 task (Modelnet40).
 
* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on a scene segmentation 
 task (S3DIS).
 
* [SLAM Segmentation](./doc/slam_segmentation_guide.md): Instructions to train KP-FCNN on a slam segmentation 
 task (SemanticKitti).
 
* [Pretrained models](./doc/pretrained_models_guide.md): We provide pretrained weights and instructions to load them.
 
* [Visualization scripts](./doc/visualization_guide.md): For now only one visualization script has been implemented: 
the kernel deformations display.

## Acknowledgment

Our code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License
Our code is released under MIT License (see LICENSE file for details).

## Updates
* 27/04/2020: Initial release.
* 27/04/2020: Added NPM3D support thanks to @GeoSur.
