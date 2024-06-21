<div align="center">

# [Object Pose Estimation via the Aggregation of Diffusion Features](https://arxiv.org/abs/2403.18791)
<h2>CVPR 2024 Highlight</h2>

Tianfu Wang, Guosheng Hu, Hongguang Wang 
</div>

<div align="justify">
Abstract: Estimating the pose of objects from images is a crucial task of 3D scene understanding, and recent approaches have shown promising results on very large benchmarks. However, these methods experience a significant performance drop when dealing with unseen objects. We believe that it results from the limited generalizability of image features. To address this problem, we have an in-depth analysis on the features of diffusion models, e.g. Stable Diffusion, which hold substantial potential for modeling unseen objects. Based on this analysis, we then innovatively introduce these diffusion features for object pose estimation. To achieve this, we propose three distinct architectures that can effectively capture and aggregate diffusion features of different granularity, greatly improving the generalizability of object pose estimation. Our approach outperforms the state-of-the-art methods by a considerable margin on three popular benchmark datasets, LM, O-LM, and T-LESS. In particular, our method achieves higher accuracy than the previous best arts on unseen objects: 98.2% vs. 93.5% on Unseen LM, 85.9% vs. 76.3% on Unseen O-LM, showing the strong generalizability of our method.
</div>

## Installation
<details><summary>Click to expand</summary>

### 1. Clone this repo.
```
git clone https://github.com/Tianfu18/diff-feats-pose.git
```
### 2. Install environments.
```
conda env create -f environment.yaml
conda activate diff-feats
```
</details>

## Data Preparation

<details><summary>Click to expand</summary>

### Final structure of folder dataset
```bash
./dataset
    ├── linemod 
        ├── models
        ├── opencv_pose
        ├── LINEMOD
        ├── occlusionLINEMOD
    ├── tless
        ├── models
        ├── opencv_pose
        ├── train
        └── test
    ├── templates	
        ├── linemod
            ├── train
            ├── test
        ├── tless
    ├── SUN397
    ├── LINEMOD.json # query-template pairwise for LINEMOD
    ├── occlusionLINEMOD.json # query-template pairwise for Occlusion-LINEMOD
    ├── tless_train.json # query-template pairwise for training split of T-LESS
    ├── tless_test.json # query-template pairwise for testing split of T-LESS
    └── crop_image512 # pre-cropped images for LINEMOD
```

### 1. Download datasets:
Download with following gdrive links and unzip them in ./dataset. We use the same data as [template-pose](https://github.com/ashawkey/stable-dreamfusion).
- [LINEMOD and Occlusion-LINEMOD (3GB)](https://drive.google.com/file/d/1XkQBt01nlfCbFuBsPMfSHlcNIzShn7e7/view?usp=sharing)
- [T-LESS (11GB)](https://drive.google.com/file/d/1d2GoswrnvcTlwFi_LWoCiy1uS5OkCiF1/view?usp=sharing)
- [SUN397, randomized background for training on T-LESS (37GB)](vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)

### 2. Process ground-truth poses
Convert the coordinate system to [BOP datasets format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) and save GT poses of each object separately:
```bash
python -m data.process_gt_linemod
python -m data.process_gt_tless
```
### 3. Render templates
To render templates:
```bash
python -m data.render_templates --dataset linemod --disable_output --num_workers 4
python -m data.render_templates --dataset tless --disable_output --num_workers 4
```
### 4. Crop images (only for LINEMOD)
Crop images of LINEMOD, OcclusionLINEMOD and its templates with GT poses:
```bash
python -m data.crop_image_linemod
```
### 5. Compute neighbors with GT poses
```bash
python -m data.create_dataframe_linemod
python -m data.create_dataframe_tless --split train
python -m data.create_dataframe_tless --split test
```

</details>

## Launch a training

<details><summary>Click to expand</summary>

### 1. Launch a training on LINEMOD
```bash
python train_linemod.py --config_path config_run/LM_Diffusion_$split_name.json
```

### 2. Launch a training on T-LESS
```bash
python train_tless.py --config_path ./config_run/TLESS_Diffusion.json
```

</details>

## Reproduce the results
<details><summary>Click to expand</summary>

### 1. Download checkpoints
You can download it from this [link](https://drive.google.com/drive/folders/1CVyW7IDAZ0uGZSJIoN3ARRyP_wY2Ntk9?usp=sharing).

### 2. Reproduce the results on LINEMOD
```bash
python test_linemod.py --config_path config_run/LM_Diffusion_$split_name.json --checkpoint checkpoint_path
```

### 3. Reproduce the results on T-LESS
```bash
python test_tless.py --config_path ./config_run/TLESS_Diffusion.json --checkpoint checkpoint_path
```
</details>

## Citation
If you find our project helpful for your research, please cite:
```bibtex
@inproceedings{wang2024object,
    title={Object Pose Estimation via the Aggregation of Diffusion Features},
    author={Wang, Tianfu and Hu, Guosheng and Wang, Hongguang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

## Acknowledgement
This codebase is built based on the [template-pose](https://github.com/nv-nguyen/template-pose). Thanks!
