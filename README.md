# Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection (ICLR 2025)

This repo is the official implementation of ICLR 2025 paper: **[Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection](https://arxiv.org/abs/2409.04796)**

> Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection
>
> Fanhu Zeng, Zhen Cheng, Fei Zhu, Hongxin Wei, Xu-Yao Zhang

[![arXiv](https://img.shields.io/badge/Arxiv-2409.04796-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.04796)
[![ICLR](https://img.shields.io/badge/OpenReview-Paper-orange.svg)](https://openreview.net/pdf?id=Ew3VifXaxZ)


## :open_book: Overview

## :sparkles: Main Results

## :rocket: Quick Start
### Environment and package 
```bash
# Create a conda environment
conda create -n localprompt python=3.8

# Activate the environment
conda activate localprompt

# clone this repo
git clone https://github.com/AuroraZengfh/Local-Prompt.git
cd Local-Prompt

# install necessary packages
pip install -r requirements.txt
```

### Install Dassl
```bash
cd Dassl

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### Data Preparation
Please create `data` folder and download the following ID and OOD datasets to `data`.

#### In-distribution Datasets
We use ImageNet-1K as the ID dataset.
- Create a folder named `imagenet/` under `data` folder.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`.

Besides, we need to put `classnames.txt` under`imagenet/data` folder. This .txt file can be downloaded via https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view

For more ID dataset like ImageNet100, ImageNet10 and ImageNet20 in the paper, follow the same procedure to construct 'classnames.txt' in the same format and refer to [MCM](https://github.com/deeplearning-wisc/MCM) to find detailed category names.

#### Out-of-distribution Datasets
We use the large-scale OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618). You can download the subsampled datasets following instructions from [this repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset).

The overall file structure of file is as follows:
```
Local-Prompt
|-- data
    |-- imagenet
        |-- classnames.txt
        |-- train/ # contains 1,000 folders like n01440764, n01443537, etc.
        |-- val/ # contains 1,000 folders like n01440764, n01443537, etc.
    |-- imagenet100
        |-- classnames.txt
        |-- train/
        |-- val/
    |-- imagenet10
        |-- classnames.txt
        |-- train/
        |-- val/
    |-- imagenet20
        |-- classnames.txt
        |-- train/
        |-- val/
    |-- iNaturalist
    |-- SUN
    |-- Places
    |-- Texture
    ...
```

### Train
The training script is in `Local-Prompt/scripts/train.sh`, you can alter the parameter in the script file.

e.g., 4-shot training with ViT-B/16
``` 
CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh data imagenet vit_b16_ep30 end 16 4 True 5 0.5 50
```

### Inference
The eval script is in Local-Prompt/scripts/train.sh, you can switch the parameter in the script file.

e.g., evaluate 4-shot model of seed1 obtained from training scripts above

```
CUDA_VISIBLE_DEVICES=0 sh scripts/eval.sh data imagenet vit_b16_ep30 16 10 output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/nctx16_cscTrue_ctpend_topk50/seed1
```

## :blue_book: Citation
If you find this work useful, consider giving this repository a star :star: and citing :bookmark_tabs: our paper as follows:

```bibtex
@inproceedings{zeng2025enhancing,
  title={Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection},
  author={Fanhu Zeng and Zhen Cheng and Fei Zhu and Hongxin Wei and Xu-Yao Zhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=Ew3VifXaxZ}
}
```

## Acknowledgememnt

The code is based on  [CoOp](https://github.com/KaiyangZhou/CoOp), [LoCoOp](https://github.com/AtsuMiyai/LoCoOp). Thanks for these great works and open sourcing! 

If you find them helpful, please consider citing them as well. 
