
## 6D Object Pose Estimation: Papers and Codes 

This repository summarizes papers and codes for **6D Object Pose Estimation** of rigid objects, which means computing the 6D transformation from the **object coordinate** to the **camera coordinate**. Let _Xo​_ represents the object's points in the object coordinate, and _Xc​_ represents the object's points in the camera coordinate, the 6D object pose ​_T_ satisfies ​_Xc = T * Xo​_ and ​_T = [R, t]​_ contains the three dimensional rotation _R​_ and the three dimensional translation ​_t​_.

Most of the current methods aim at **instance-level 6D object pose estimation**, which means that the identical 3D model exists. There also emerges **category-level 6D object pose estimation**, which means that the observed object could be not identical to existing 3D models but come from a same geometric category. Based on the inputs, methods can also be categorized into __RGB-D image-based methods__ and __point cloud-based methods__. 

# Table of Contents
1. [Instance-level 6D Object Pose Estimation](#instance-level)

&nbsp;&nbsp;&nbsp;&nbsp;1.1 [RGB-D Image-based Methods](#instance-rgb)
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.1. [Correspondence-based Methods](#correspondance-based)
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. [Match 2D feature points](#match-2d)
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. [Regress 2D projections](#regress-2d)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.2. [Template-based Methods](#template-based)
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.3. [Voting-based Methods](#voting-based)
    
2. [Category-level Methods](#category-level)

&nbsp;&nbsp;&nbsp;&nbsp;2.1. [Category-level 6D pose estimation](#category-6d)
  
&nbsp;&nbsp;&nbsp;&nbsp;2.2. [3D shape reconstruction from images](#category-3d)
  
&nbsp;&nbsp;&nbsp;&nbsp;2.3. [3D shape rendering](#3d-shape)
  
  
### 1. Instance-level 6D Object Pose Estimation <a name="instance-level"></a>

#### 1.1 RGB-D Image-based Methods <a name="instance-rgb"></a>

***Survey papers:***

**[arXiv]** A Review on Object Pose Recovery: from 3D Bounding Box Detectors to Full 6D Pose Estimators, [[paper](https://arxiv.org/pdf/2001.10609.pdf)]

***2016:***

**[ECCVW]** A Summary of the 4th International Workshop on Recovering 6D Object Pose, [[paper](https://arxiv.org/abs/1810.03758)]

#### 1.1.1 Correspondence-based Methods <a name="correspondance-based"></a>

##### a. Match 2D feature points <a name="match-2d"></a>

***2020:***

**[arXiv]** Delta Descriptors: Change-Based Place Representation for Robust Visual Localization, [[paper](https://arxiv.org/pdf/2006.05700.pdf)]

**[arXiv]** Unconstrained Matching of 2D and 3D Descriptors for 6-DOF Pose Estimation, [[paper](https://arxiv.org/pdf/2005.14502.pdf)]

**[arXiv]** S2DNet: Learning Accurate Correspondences for Sparse-to-Dense Feature Matching, [[paper](https://arxiv.org/pdf/2004.01673.pdf)]

**[arXiv]** SK-Net: Deep Learning on Point Cloud via End-to-end Discovery of Spatial Keypoints, [[paper](https://arxiv.org/pdf/2003.14014.pdf)]

**[arXiv]** LRC-Net: Learning Discriminative Features on Point Clouds by Encoding Local Region Contexts, [[paper](https://arxiv.org/pdf/2003.08240.pdf)]

**[arXiv]** Table-Top Scene Analysis Using Knowledge-Supervised MCMC, [[paper](https://arxiv.org/pdf/2002.08417.pdf)]

**[arXiv]** AprilTags 3D: Dynamic Fiducial Markers for Robust Pose Estimation in Highly Reflective Environments and Indirect Communication in Swarm Robotics, [[paper](https://arxiv.org/pdf/2001.08622.pdf)]

**[AAAI]** LCD: Learned Cross-Domain Descriptors for 2D-3D Matching, [[paper](https://arxiv.org/abs/1911.09326)] [[project](https://hkust-vgd.github.io/lcd/)]

***2019:***

**[ICCV]** GLAMpoints: Greedily Learned Accurate Match points, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Truong_GLAMpoints_Greedily_Learned_Accurate_Match_Points_ICCV_2019_paper.pdf)]

***2016:***

**[ECCV]** LIFT: Learned Invariant Feature Transform, [[paper](https://arxiv.org/pdf/1603.09114.pdf)]

***2012:***

**[3DIMPVT]** 3D Object Detection and Localization using Multimodal Point Pair Features, [[paper](http://far.in.tum.de/pub/drost20123dimpvt/drost20123dimpvt.pdf)]



##### b. Regress 2D projections <a name="regress-2d"></a>

***2020:***

**[arXiv]** PrimA6D: Rotational Primitive Reconstruction for Enhanced and Robust 6D Pose Estimation, [[paper](https://arxiv.org/pdf/2006.07789.pdf)]

**[arXiv]** EPOS: Estimating 6D Pose of Objects with Symmetries, [[paper](https://arxiv.org/pdf/2004.00605.pdf)]

**[arXiv]** Tackling Two Challenges of 6D Object Pose Estimation: Lack of Real Annotated RGB Images and Scalability to Number of Objects, [[paper](https://arxiv.org/pdf/2003.12344.pdf)]

**[arXiv]** Squeezed Deep 6DoF Object Detection using Knowledge Distillation, [[paper](https://arxiv.org/pdf/2003.13586.pdf)]

**[arXiv]** Learning 2D–3D Correspondences To Solve The Blind Perspective-n-Point Problem, [[paper](https://arxiv.org/pdf/2003.06752.pdf)]

**[arXiv]** PnP-Net: A hybrid Perspective-n-Point Network, [[paper](https://arxiv.org/pdf/2003.04626.pdf)]

**[arXiv]** Object 6D Pose Estimation with Non-local Attention, [[paper](https://arxiv.org/pdf/2002.08749.pdf)]

**[arXiv]** 6DoF Object Pose Estimation via Differentiable Proxy Voting Loss, [[paper](https://arxiv.org/pdf/2002.03923.pdf)]

***2019:***

**[arXiv]** DPOD: 6D Pose Object Detector and Refiner, [[paper](https://arxiv.org/pdf/1902.11020.pdf)]

**[CVPR]** Segmentation-driven 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1812.02541)] [[code](https://github.com/cvlab-epfl/segmentation-driven-pose)]

**[arXiv]** Single-Stage 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1911.08324)]

**[arXiv]** W-PoseNet: Dense Correspondence Regularized Pixel Pair Pose Regression, [[paper](https://arxiv.org/pdf/1912.11888.pdf)]

**[arXiv]** KeyPose: Multi-view 3D Labeling and Keypoint Estimation for Transparent Objects, [[paper](https://arxiv.org/abs/1912.02805)]

***2018:***

**[CVPR]** Real-time seamless single shot 6d object pose prediction, [[paper](https://arxiv.org/abs/1711.08848)] [[code](https://github.com/Microsoft/singleshotpose)]

**[arXiv]** Estimating 6D Pose From Localizing Designated Surface Keypoints, [[paper](https://arxiv.org/abs/1812.01387)]

***2017:***

**[ICCV]** BB8: a scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth, [[paper](https://arxiv.org/abs/1703.10896)]

**[ICCV]** SSD-6D: Making rgb-based 3d detection and 6d pose estimation great again, [[paper](https://arxiv.org/abs/1711.10006)] [[code](https://github.com/wadimkehl/ssd-6d)]

**[ICRA]** 6-DoF Object Pose from Semantic Keypoints, [[paper](https://arxiv.org/abs/1703.04670)]



#### 1.1.2 Template-based Methods <a name="template-based"></a>

This kind of methods can be regarded as regression-based methods.

***2020:***

**[arXiv]** A survey on deep supervised hashing methods for image retrieval, [[paper](https://arxiv.org/pdf/2006.05627.pdf)]

**[arXiv]** Neural Object Learning for 6D Pose Estimation Using a Few Cluttered Images, [[paper](https://arxiv.org/pdf/2005.03717.pdf)]

**[arXiv]** How to track your dragon: A Multi-Attentional Framework for real-time RGB-D 6-DOF Object Pose Tracking, [[paper](https://arxiv.org/pdf/2004.10335.pdf)]

**[arXiv]** Self6D: Self-Supervised Monocular 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.06468.pdf)]

**[arXiv]** A Novel Pose Proposal Network and Refinement Pipeline for Better Object Pose Estimation, [[paper](https://arxiv.org/pdf/2004.05507.pdf)]

**[arXiv]** G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features, [[paper](https://arxiv.org/pdf/2003.11089.pdf)] [[code](https://github.com/DC1991/G2L_Net)]

**[arXiv]** Neural Mesh Refiner for 6-DoF Pose Estimation, [[paper](https://arxiv.org/pdf/2003.07561.pdf)]

**[arXiv]** MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision, [[paper](https://arxiv.org/pdf/2003.03522.pdf)]

**[arXiv]** Robust 6D Object Pose Estimation by Learning RGB-D Features, [[paper](https://arxiv.org/pdf/2003.00188.pdf)]

**[arXiv]** HybridPose: 6D Object Pose Estimation under Hybrid Representations, [[paper](https://arxiv.org/pdf/2001.01869.pdf)] [[code](https://github.com/chensong1995/HybridPose)]

***2019:***

**[arXiv]** P<sup>2</sup>GNet: Pose-Guided Point Cloud Generating Networks for 6-DoF Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.09316)]

**[arXiv]** ConvPoseCNN: Dense Convolutional 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.07333)]

**[arXiv]** PointPoseNet: Accurate Object Detection and 6 DoF Pose Estimation in Point Clouds, [[paper](https://arxiv.org/abs/1912.09057)]

**[RSS]** PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking, [[paper](https://arxiv.org/abs/1905.09304)]

**[arXiv]** Multi-View Matching Network for 6D Pose Estimation, [[paper](https://arxiv.org/abs/1911.12330)]

**[arXiv]** Fast 3D Pose Refinement with RGB Images, [[paper](https://arxiv.org/pdf/1911.07347.pdf)]

**[arXiv]** MaskedFusion: Mask-based 6D Object Pose Detection, [[paper](https://arxiv.org/abs/1911.07771)]

**[CoRL]** Scene-level Pose Estimation for Multiple Instances of Densely Packed Objects, [[paper](https://arxiv.org/abs/1910.04953)]

**[IROS]** Learning to Estimate Pose and Shape of Hand-Held Objects from RGB Images, [[paper](https://arxiv.org/abs/1903.03340)]

**[IROSW]** Motion-Nets: 6D Tracking of Unknown Objects in Unseen Environments using RGB, [[paper](https://arxiv.org/abs/1910.13942)]

**[ICCV]** DPOD: 6D Pose Object Detector and Refiner, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)]

**[ICCV]** CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.html)] [[code](https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV)]

**[ICCV]** Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Pix2Pose_Pixel-Wise_Coordinate_Regression_of_Objects_for_6D_Pose_Estimation_ICCV_2019_paper.pdf)] [[code](https://github.com/kirumang/Pix2Pose)]

**[ICCV]** Explaining the Ambiguity of Object Detection and 6D Pose From Visual Data, [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Manhardt_Explaining_the_Ambiguity_of_Object_Detection_and_6D_Pose_From_ICCV_2019_paper.pdf)]

**[arXiv]** Active 6D Multi-Object Pose Estimation in Cluttered Scenarios with Deep Reinforcement Learning, [[paper](https://arxiv.org/abs/1910.08811)]

**[arXiv]** Accurate 6D Object Pose Estimation by Pose Conditioned Mesh Reconstruction, [[paper](https://arxiv.org/abs/1910.10653)]

**[arXiv]** Learning Object Localization and 6D Pose Estimation from Simulation and Weakly Labeled Real Images, [[paper](https://arxiv.org/abs/1806.06888)]

**[ICHR]** Refining 6D Object Pose Predictions using Abstract Render-and-Compare, [[paper](https://arxiv.org/abs/1910.03412)]

**[arXiv]** Deep-6dpose: recovering 6d object pose from a single rgb image, [[paper](https://arxiv.org/abs/1901.04780)]

**[arXiv]** Real-time Background-aware 3D Textureless Object Pose Estimation, [[paper](https://arxiv.org/abs/1907.09128)]

**[IROS]** SilhoNet: An RGB Method for 6D Object Pose Estimation, [[paper](https://arxiv.org/abs/1809.06893)]

***2018:***

**[ECCV]** AAE: Implicit 3D Orientation Learning for 6D Object Detection From RGB Images, [[paper](https://arxiv.org/abs/1902.01275)] [[code](https://github.com/DLR-RM/AugmentedAutoencoder)]

**[ECCV]** DeepIM:Deep Iterative Matching for 6D Pose Estimation, [[paper](https://arxiv.org/abs/1804.00175)] [[code](https://github.com/liyi14/mx-DeepIM)]

**[RSS]** Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes, [[paper](https://arxiv.org/abs/1711.00199)] [[code](https://github.com/yuxng/PoseCNN)]

**[IROS]** Robust 6D Object Pose Estimation in Cluttered Scenes using Semantic Segmentation and Pose Regression Networks, [[paper](https://arxiv.org/abs/1810.03410)]

***2012:***

**[ACCV]** Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.250.6547&rep=rep1&type=pdf)]



#### 1.1.3 Voting-based Methods <a name="voting-based"></a>

***2019:***

**[CVPR]** PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation, [[paper](https://arxiv.org/abs/1812.11788)] [[code](https://github.com/zju3dv/pvnet)]

***2017:***

**[TPAMI]** Robust 3D Object Tracking from Monocular Images Using Stable Parts, [[paper](https://ieeexplore.ieee.org/document/7934426)]

**[Access]** Fast Object Pose Estimation Using Adaptive Threshold for Bin-picking, [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046779)]

***2014:***

**[ECCV]** Learning 6d object pose estimation using 3d object coordinate, [[paper](http://wwwpub.zih.tu-dresden.de/~cvweb/publications/papers/2014/PoseEstimationECCV2014.pdf)]

**[ECCV]** Latent-class hough forests for 3d object detection and pose estimation, [[paper](https://labicvl.github.io/docs/pubs/Aly_ECCV_2014.pdf)]



***Datasets:*** <a name="1-1-3-datasets"></a>

[LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, ACCV, 2012 [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.250.6547&rep=rep1&type=pdf)] [[database](https://github.com/paroj/linemod_dataset)]

[YCB Datasets](http://www.ycbbenchmarks.com): The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, IEEE International Conference on Advanced Robotics (ICAR), 2015 [[paper](http://dx.doi.org/10.1109/ICAR.2015.7251504)]

[T-LESS Datasets](http://cmp.felk.cvut.cz/t-less/): T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, IEEE Winter Conference on Applications of Computer Vision (WACV), 2017 [[paper](https://arxiv.org/abs/1701.05498)]

HomebrewedDB: RGB-D Dataset for 6D Pose Estimation of 3D Objects, ICCVW, 2019 [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Kaskman_HomebrewedDB_RGB-D_Dataset_for_6D_Pose_Estimation_of_3D_Objects_ICCVW_2019_paper.pdf)]

YCB-M: A Multi-Camera RGB-D Dataset for Object Recognition and 6DoF Pose Estimation, arXiv, 2020, [[paper](https://arxiv.org/pdf/2004.11657.pdf)] [[database](https://zenodo.org/record/2579173#.XqgpkxMzbX8)]



### 2 Category-level Methods <a name="category-level"></a>

#### 2.1 Category-level 6D pose estimation <a name="category-6d"></a>

***2020:***

**[arXiv]** CPS: Class-level 6D Pose and Shape Estimation From Monocular Images, [[paper](https://arxiv.org/pdf/2003.05848v1.pdf)]

**[arXiv]** Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation, [[paper](https://arxiv.org/pdf/2001.09322.pdf)]

***2019:***

**[arXiv]** Category-Level Articulated Object Pose Estimation, [[paper](https://arxiv.org/pdf/1912.11913.pdf)]

**[arXiv]** LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation, [[paper](https://arxiv.org/abs/1912.00416)]

**[arXiv]** 6-PACK: Category-level 6D Pose Tracker with Anchor-Based Keypoints, [[paper](https://arxiv.org/abs/1910.10750)] [[code](https://github.com/j96w/6-PACK)]

**[arXiv]** Self-Supervised 3D Keypoint Learning for Ego-motion Estimation, [[paper](https://arxiv.org/abs/1912.03426)]

**[CVPR]** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, [[paper](https://arxiv.org/abs/1901.02970)] [[code](https://github.com/hughw19/NOCS_CVPR2019)] 

**[arXiv]** Instance- and Category-level 6D Object Pose Estimation, [[paper](https://arxiv.org/pdf/1903.04229.pdf)]

**[arXiv]** kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation, [[paper](https://arxiv.org/abs/1903.06684)]



#### 2.2 3D shape reconstruction from images <a name="category-3d"></a>

***2020:***

**[arXiv]** Joint Hand-object 3D Reconstruction from a Single Image with Cross-branch Feature Fusion, [[paper](https://arxiv.org/pdf/2006.15561.pdf)]

**[arXiv]** Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images, [[paper](https://arxiv.org/pdf/2006.12250.pdf)]

**[arXiv]** 3D Shape Reconstruction from Free-Hand Sketches, [[paper](https://arxiv.org/pdf/2006.09694.pdf)]

**[arXiv]** Learning to Detect 3D Reflection Symmetry for Single-View Reconstruction, [[paper](https://arxiv.org/pdf/2006.10042.pdf)]

**[arXiv]** Convolutional Generation of Textured 3D Meshes, [[paper](https://arxiv.org/pdf/2006.07660.pdf)]

**[arXiv]** 3D Reconstruction of Novel Object Shapes from Single Images, [[paper](https://arxiv.org/pdf/2006.07752.pdf)]

**[arXiv]** Novel Object Viewpoint Estimation through Reconstruction Alignment, [[paper](https://arxiv.org/pdf/2006.03586.pdf)]

**[arXiv]** UCLID-Net: Single View Reconstruction in Object Space, [[paper](https://arxiv.org/pdf/2006.03817.pdf)]

**[arXiv]** SurfaceNet+: An End-to-end 3D Neural Network for Very Sparse Multi-view Stereopsis, [[paper](https://arxiv.org/pdf/2005.12690.pdf)]

**[arXiv]** FroDO: From Detections to 3D Objects, [[paper](https://arxiv.org/pdf/2005.05125.pdf)]

**[arXiv]** CoReNet: Coherent 3D scene reconstruction from a single RGB image, [[paper](https://arxiv.org/pdf/2004.12989.pdf)]

**[arXiv]** Reconstruct, Rasterize and Backprop: Dense shape and pose estimation from a single image, [[paper](https://arxiv.org/pdf/2004.12232.pdf)]

**[arXiv]** Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes, [[paper](https://arxiv.org/pdf/2004.10904.pdf)]

**[arXiv]** Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors, [[paper](https://arxiv.org/pdf/2004.06302.pdf)]

**[arXiv]** Neural Object Descriptors for Multi-View Shape Reconstruction, [[paper](https://arxiv.org/pdf/2004.04485.pdf)]

**[arXiv]** Leveraging 2D Data to Learn Textured 3D Mesh Generation, [[paper](https://arxiv.org/pdf/2004.04180.pdf)]

**[arXiv]** Deep 3D Capture: Geometry and Reflectance from Sparse Multi-View Images, [[paper](https://arxiv.org/pdf/2003.12642.pdf)]

**[arXiv]** Self-Supervised 2D Image to 3D Shape Translation with Disentangled Representations, [[paper](https://arxiv.org/pdf/2003.10016.pdf)]

**[arXiv]** Atlas: End-to-End 3D Scene Reconstruction from Posed Images, [[paper](https://arxiv.org/pdf/2003.10432.pdf)]

**[arXiv]** Instant recovery of shape from spectrum via latent space connections, [[paper](https://arxiv.org/pdf/2003.06523.pdf)]

**[arXiv]** Self-supervised Single-view 3D Reconstruction via Semantic Consistency, [[paper](https://arxiv.org/pdf/2003.06473.pdf)]

**[arXiv]** Meta3D: Single-View 3D Object Reconstruction from Shape Priors in Memory, [[paper](https://arxiv.org/pdf/2003.03711.pdf)]

**[arXiv]** STD-Net: Structure-preserving and Topology-adaptive Deformation Network for 3D Reconstruction from a Single Image, [[paper](https://arxiv.org/pdf/2003.03551.pdf)]

**[arXiv]** Inverse Graphics GAN: Learning to Generate 3D Shapes from Unstructured 2D Data, [[paper](https://arxiv.org/pdf/2002.12674.pdf)]

**[arXiv]** Deep NRSfM++: Towards 3D Reconstruction in the Wild, [[paper](https://arxiv.org/pdf/2001.10090.pdf)]

**[arXiv]** Learning to Correct 3D Reconstructions from Multiple Views, [[paper](https://arxiv.org/pdf/2001.08098.pdf)]

***2019:***

**[arXiv]** Boundary Cues for 3D Object Shape Recovery, [[paper](https://arxiv.org/pdf/1912.11566.pdf)]

**[arXiv]** Learning to Generate Dense Point Clouds with Textures on Multiple Categories, [[paper](https://arxiv.org/pdf/1912.10545.pdf)]

**[arXiv]** Front2Back: Single View 3D Shape Reconstruction via Front to Back Prediction, [[paper](https://arxiv.org/pdf/1912.10589.pdf)]

**[arXiv]** Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision, [[paper](https://arxiv.org/abs/1912.07372)]

**[arXiv]** SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization, [[paper](https://arxiv.org/abs/1912.07109)]

**[arXiv]** 3D-GMNet: Learning to Estimate 3D Shape from A Single Image As A Gaussian Mixture, [[paper](https://arxiv.org/abs/1912.04663)]

**[arXiv]** Deep-Learning Assisted High-Resolution Binocular Stereo Depth Reconstruction, [[paper](https://arxiv.org/abs/1912.05012)]


2.2 3D shape reconstruction from images <a name="category-3d"></a>
#### 2.3 3D shape rendering <a name="3d-shape"></a>

***2020:***

**[arXiv]** Intrinsic Autoencoders for Joint Neural Rendering and Intrinsic Image Decomposition, [[paper](https://arxiv.org/pdf/2006.16011.pdf)]

**[arXiv]** SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans, [[paper](https://arxiv.org/pdf/2006.14660.pdf)]

**[arXiv]** Differentiable Rendering: A Survey, [[paper](https://arxiv.org/pdf/2006.12057.pdf)]

**[arXiv]** Equivariant Neural Rendering, [[paper](https://arxiv.org/pdf/2006.07630.pdf)]

***2019:***

**[arXiv]** SynSin: End-to-end View Synthesis from a Single Image, [[paper](https://arxiv.org/abs/1912.08804)] [[project](http://www.robots.ox.ac.uk/~ow/synsin.html)]

**[arXiv]** Neural Point Cloud Rendering via Multi-Plane Projection, [[paper](https://arxiv.org/abs/1912.04645)]

**[arXiv]** Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool, [[paper](https://arxiv.org/abs/1912.04591)]

