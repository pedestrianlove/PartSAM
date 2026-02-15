<div align="center">

# PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data

</div>

<div align="center">

[Zhe Zhu](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=zh-CN)<sup>1</sup>, [Le Wan](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=zh-CN)<sup>2</sup>, [Rui Xu](https://ruixu.me/)<sup>3</sup>, [Yiheng Zhang](https://openreview.net/profile?id=~Yiheng_Zhang4)<sup>4</sup>, [Honghua Chen](https://chenhonghua.github.io/clay.github.io/)<sup>5</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>3</sup>, [Cheng Lin](https://clinplayer.github.io/)<sup>6</sup>, [Yuan Liu](https://liuyuan-pal.github.io/)<sup>2&dagger;</sup>, [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ&hl=en)<sup>1&dagger;</sup>
<br>
&dagger; Corresponding authors


<sup>1</sup> NUAA
<sup>2</sup> HKUST
<sup>3</sup> HKU
<sup>4</sup> NUS
<sup>5</sup> LU
<sup>6</sup> MUST


<p align="center">
  <a href="https://czvvd.github.io/PartSAMPage/">
    <img src="https://img.shields.io/badge/Project%20Page-blue.svg" alt="Project Page" height="22">
  </a>
  <a href="https://arxiv.org/abs/2509.21965">
      <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?logo=arXiv&logoColor=white" alt="arXiv height="22">
  </a>
</p>
</div>

<p align="center">
  <img src="assets/teaser.png" alt="teaser">
</p>



## Installation
1. Install the required environment
```
conda create -n PartSAM python=3.11 -y
conda activate PartSAM
# PyTorch 2.4.1 with CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d safetensors
pip install hydra-core omegaconf accelerate timm igraph ninja
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
apt install libx11-6 libgl1 libxrender1
pip install vtk
```

2. Install other third-party modules (torkit3d and apex) following [Point-SAM](https://github.com/zyc00/Point-SAM)

3. Install the pretrained model weight
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download Czvvd/PartSAM --local-dir ./pretrained
```

## Usage
```
# Modify the config file to evaluate your own meshes
python evaluation/eval_everypart.py
```

## TODO
- [x] Release inference code of PartSAM
- [x] Release the pre-trained models
- [ ] Release training code and data processing script

## Acknowledgement
Our code is based on these wonderful works:
* [Point-SAM](https://github.com/zyc00/Point-SAM)
* [PartField](https://github.com/nv-tlabs/PartField)
* [SAMPart3D](https://github.com/Pointcept/SAMPart3D)
* [SAMesh](https://github.com/gtangg12/samesh)
* [Find3D](https://github.com/ziqi-ma/Find3D)

We thank the authors for their great work！



## Citation

```bibtex
@article{zhu2025partsam,
  title={PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data}, 
  author={Zhe Zhu and Le Wan and Rui Xu and Yiheng Zhang and Honghua Chen and Zhiyang Dou and Cheng Lin and Yuan Liu and Mingqiang Wei},
  journal={arXiv preprint arXiv:2509.21965},
  year={2025}
}
```
