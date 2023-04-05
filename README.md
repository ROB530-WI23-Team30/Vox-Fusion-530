# ROB530-Project
project repo for ROB530 WN 2023

## Tasks and Milestones
+ find/create map data that contains illumination change
  + 2/3 people
  + deadline: 3.13 (spring break ends)
+ concrete implementation idea
  + 2/3 people
  + deadline: 3.6 (concurrent)
+ project implementation
  + all people
  + more detailed later
  + start: 3.6
  + deadline: 4.10
+ experiments & evaluation
  + 2/3 people
  + start: 4.3
  + end: 4.14
+ paper writing & video making
  + all people
  + start: 4.3
  + end: 4.18

## Install

+ Install [poetry](https://python-poetry.org/docs/)
+ disable poetry to use keyring

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

+ prepare a python3.8 and set poetry environment using

```bash
poetry env use /path/to/python3.8
```

+ install

```bash
poetry install
```

+ entering environment

```bash
poetry shell
```

+ prepare pre-commit hook

```bash
pre-commit install
```


## Dataset
### ETHL

Download [here](http://cvg.ethz.ch/research/illumination-change-robust-dslam/).

+ TUM compatible dataset

+ Synthetic data is 30FPS
+ https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html for intrinsics
+ https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats for file formats

### ETH3D

Download [here](https://www.eth3d.net/slam_datasets). Some datasets contain illumination changes.
