# CADDY: Continuous-time Anomaly Detection in Dynamic Networks

![framework](https://github.com/Jie-0828/CADDY/assets/105060483/9c53fa98-7d85-4c49-9a24-4c11cf699764)

## Requirments
* Python==3.8
* PyTorch==1.7.1
* Transformers==3.5.1
* Scipy==1.5.2
* Numpy==1.19.2
* Networkx==2.5
* Scikit-learn==0.23.2

## Usage
### Step 0: Prepare Data
```
python 0_prepare_data.py --dataset uci
```

### Step 1: Train Model
```
python 1_train.py --dataset uci --anomaly_per 0.1
