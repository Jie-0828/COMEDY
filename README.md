# CADDY: Continuous-time Anomaly Detection in Dynamic Networks
<!--#### -->
## Introduction
![framework](https://github.com/Jie-0828/CADDY/assets/105060483/9c53fa98-7d85-4c49-9a24-4c11cf699764)

## Dataset and preprocessing

### Download the public dataset
* [HDFS](https://doi.org/10.5281/zenodo.1144100)
  
* [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
  
* [Brightkite](http://snap.stanford.edu/data/loc-brightkite.html)

## Usage
### Step 0: Prepare Data
```
python 0_prepare_data.py --dataset uci
```

### Step 1: Train Model
```
python 1_train.py --dataset uci --anomaly_per 0.1


## Command and configurations
### General flags

## Requirements
* python >= 3.9

* Dependency

```{bash}
torch==1.12.0
networkx==2.7.1
tqdm==4.64.0
numpy==1.25.5
scikit-learn==1.0.2
random==
```
