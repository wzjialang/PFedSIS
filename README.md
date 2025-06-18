# [IEEE TBME Featured Article] PFedSIS: Personalizing Federated Instrument Segmentation With Visual Trait Priors in Robotic Surgery

This repository contains the official PyTorch implementation of the paper:
> [**Personalizing Federated Instrument Segmentation With Visual Trait Priors in Robotic Surgery**](https://doi.org/10.1109/LRA.2024.3505818)<br>
> **Authors:** [Jialang Xu](https://www.linkedin.com/in/jialang-xu-778952257/), Jiacheng Wang, Lequan Yu, Danail Stoyanov, Yueming Jin, Evangelos B. Mazomenos


## Abstract
Personalized federated learning (PFL) for surgical instrument segmentation (SIS) is a promising approach. It enables multiple clinical sites to collaboratively train a series of models in privacy, with each model tailored to the individual distribution of each site. Existing PFL methods rarely consider the personalization of multi-headed self-attention, and do not account for appearance diversity and instrument shape similarity, both inherent in surgical scenes. We thus propose PFedSIS, a novel PFL method with visual trait priors for SIS, incorporating global-personalized disentanglement (GPD), appearance-regulation personalized enhancement (APE), and shape-similarity global enhancement (SGE), to boost SIS performance in each site. GPD represents the first attempt at head-wise assignment for multi-headed self-attention personalization. To preserve the unique appearance representation of each site and gradually leverage the inter-site difference, APE introduces appearance regulation and provides customized layer-wise aggregation solutions via hypernetworks for each site's personalized parameters. The mutual shape information of instruments is maintained and shared via SGE, which enhances the cross-style shape consistency on the image level and computes the shape-similarity contribution of each site on the prediction level for updating the global parameters. PFedSIS outperforms state-of-the-art methods with +1.51% Dice, +2.11% IoU, −2.79 ASSD, −15.55 HD95 performance gains.

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/wzjialang/PFedSIS.git
```

### 2. Install Dependencies
Ensure the following dependencies are installed:
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

Example installation:
```bash
conda create -n pfedsis python=3.9
conda activate pfedsis
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
cd PFedSIS
pip install -r requirement.txt
```

### 3. Prepare Datasets
Download the EndoVis 2017 (Site-1), EndoVis 2018 (Site-2), and SAR-RARP (Site-3) datasets [HERE](http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2) and unzip them under the `PFedSIS` folder:
```bash
unzip robotics_site_npy.zip -d PFedSIS/
```

After unzipping, the folder structure should look like:
```
PFedSIS
├─ dataloaders/           # Dataset loaders and augmentations
├─ net/                   # Network architectures
├─ pretrained/            # PVT pre-trained weights
├─ robotics_site_npy/     # EndoVis & SAR-RARP datasets
├─ scripts/               # Training and testing utilities
├─ utils/                 # Logging, hypernetwork utilities, etc.
└─ weight/                # PFedSIS pre-trained weights
   ├─ Overall_Site{}_best.pth  # Best weights among three sites from the same epoch
   ├─ Site{}_best.pth          # Best weights for each site (possibly from different epochs)
```

## Usage
### Inference and Visualization
```bash
python inference.py --load_path ./weight
```
- Logs and visualization results will be saved in the folder specified by `--load_path`.

### Training
```bash
python train.py --exp your_experiment_name
```
- Logs and experimental results will be saved automatically under the `PFedSIS/robotics` folder.

## Acknowledgments
We sincerely appreciate the authors for releasing the following valuable resources: [lightas/FedSeg](https://github.com/lightas/FedSeg), [KarhouTam/pFedLA](https://github.com/KarhouTam/pFedLA), [jcwang123/PFL-Seg-Trans](https://github.com/jcwang123/PFL-Seg-Trans), [whai362/PVT](https://github.com/whai362/PVT), [jingyzhang/S3R](https://github.com/jingyzhang/S3R)

## Citation
If you find this project useful, please consider citing:
```bibtex
@ARTICLE{pfedsis,
  author={Xu, Jialang and Wang, Jiacheng and Yu, Lequan and Stoyanov, Danail and Jin, Yueming and Mazomenos, Evangelos B.},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={Personalizing Federated Instrument Segmentation With Visual Trait Priors in Robotic Surgery}, 
  year={2025},
  volume={72},
  number={6},
  pages={1886-1896},
  keywords={Instruments;Shape;Surgery;Training;Servers;Federated learning;Data models;Biomedical engineering;Visualization;Regulation;Personalized federated learning;multi-headed self-attention;hypernetwork;appearance regulation;shape similarity},
  doi={10.1109/TBME.2025.3526667}}
```
