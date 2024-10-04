# QQB: QuickQuakeBuildings

This repo contains dataset, code, and explanation of the paper: [QuickQuakeBuildings: Post-earthquake SAR-Optical Dataset for Quick Damaged-building Detection](https://ieeexplore.ieee.org/document/10542156) by Yao Sun, Yi Wang, and Michael Eineder. 

👉 Check out the [LinkedIn Article](https://www.linkedin.com/pulse/introducing-quickquakebuildings-new-dataset-rapid-building-yao-sun-md0jf/?trackingId=k8J2cwO9T%2FiEJUfcYFWLbQ%3D%3D) for the background and motivation. 

## Introduction

Quick and automated earthquake-damaged building detection from post-event satellite imagery is crucial, yet it is challenging due to the scarcity of training data required to develop robust algorithms. In this work, we provide the first dataset dedicated to detecting earthquake-damaged buildings from post-event very high resolution (VHR) Synthetic Aperture Radar (SAR) and optical imagery. 

Utilizing open satellite imagery and annotations acquired after the 2023 Turkey-Syria earthquakes, we deliver a dataset of coregistered building footprints and satellite image patches of both SAR and optical data, encompassing more than four thousand buildings. The task of damaged building detection is formulated as a binary image classification problem, that can also be treated as an anomaly detection problem due to extreme class imbalance. We provide baseline methods and results to serve as references for comparison. 

## Datasource

### Satellite imagery

- SAR

	The SAR image was obtained from [Capella Space Synthetic Aperture Radar Open Dataset](https://www.capellaspace.com/gallery/). 

	The used image is of type Geocoded Terrain Corrected (GEO), and downloaded from [here](https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-datetime/capella-open-data-2023/capella-open-data-2023-02/capella-open-data-2023-02-09/CAPELLA_C08_SP_GEO_HH_20230209073421_20230209073446/CAPELLA_C08_SP_GEO_HH_20230209073421_20230209073446.json).

- Optical

	The optical image was obtained from Maxar Analysis-Ready Data (ARD) under [Maxar’s open data program](https://www.maxar.com/open-data/turkey-earthquake-2023). The CATALOG ID and QUAD KEY of the used imagery are listed below.

	| CATALOG ID      | QUAD KEY       |
	|-----------------|----------------|
	|1040010082698700 | 031133012123   |
	|                 | 031133012132   |
	|                 | 031133012301   |
	|                 | 031133012310   |


### Building footprints and labels of destroyed buildings

Post-event building footprints and labels of destroyed buildings were obtained from OpenStreetMap and Humanitarian OpenStreetMap Team:

- [Turkey Buildings](https://data.humdata.org/dataset/hotosm_tur_buildings)

- [HOTOSM Turkey Destroyed Buildings](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings)

	> This theme includes all OpenStreetMap features in this area matching: destroyed:building = 'yes' AND damage:date = '2023-02-06'

## Usage

Downaload the dataset from [here](https://terabox.com/s/1LFynV38hkF2-xEAJwPGM8w), and then extract the dataset to `./data`. The dataset should be organized as follows:
```
data
├── damaged
│   ├── OSMID_SAR.mat
│   ├── OSMID_SARftp.mat
│   ├── OSMID_opt.mat
│   ├── OSMID_optftp.mat
└── |── ...
│   intact
│   ├── ...
└── fold-1.txt
└── fold-2.txt
└── fold-3.txt
└── fold-4.txt
└── fold-5.txt
```

We use ImageNet weights for optical images and footprints, and [SAR-HUB](https://github.com/XAI4SAR/SAR-HUB) weights for SAR imagery, which were pretrained on TerraSAR-X data. You can download the weights [here](https://drive.google.com/file/d/1JgCQIXMFYbTBhGbXCb1nXlLC62Ahv9qW/view?usp=drive_link).

Depending on the training mode, run:

```
# sar + opt
python main.py \
--root ./data \
--val_split fold-1.txt \
--mode all \ # choose from [all, sar, opt]
--checkpoints checkpoints_all/rn18_pretrain_fold1 \
--sar_pretrain ./weights/ResNet18_TSX.pth \
--opt_pretrain imagenet \
```

```
# sar
python main.py \
--root ./data \
--val_split fold-1.txt \
--mode sar \ # choose from [all, sar, opt]
--checkpoints checkpoints_all/rn18_pretrain_fold1 \
--sar_pretrain ./weights/ResNet18_TSX.pth \
```

```
# opt
python main.py \
--root ./data \
--val_split fold-1.txt \
--mode opt \ # choose from [all, sar, opt]
--checkpoints checkpoints_all/rn18_pretrain_fold1 \
--opt_pretrain imagenet \
```

## Results 

![Benchmark results on the dataset.](/images/table1.png)

![Examples of prediction results using SAR, optical, and both data, from their corresponding best model.](/images/visualization.png)

## Acknowledgment

We would like to thank [Capella Space](https://www.capellaspace.com/) and [Maxar Technologies](https://www.maxar.com/) for providing satellite imagery under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), and [OpenStreetMap](https://www.openstreetmap.org) and [Humanitarian OpenStreetMap Team](https://www.hotosm.org/) for providing building footprints and annotations of destroyed buildings under [ODbL License](https://opendatacommons.org/licenses/odbl/1.0/). 


## Citation

If you find the repo useful, please consider cite the following paper:
```
@article{sun2023qqb,
  author={Sun, Yao and Wang, Yi and Eineder, Michael},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={QuickQuakeBuildings: Post-Earthquake SAR-Optical Dataset for Quick Damaged-Building Detection}, 
  year={2024},
  volume={21},
  pages={1-5},
  doi={10.1109/LGRS.2024.3406966}}

```
