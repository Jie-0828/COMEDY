# CADDY: Continuous-time Anomaly Detection in Dynamic Networks
<!--#### -->
## Introduction
![framework](https://github.com/Jie-0828/CADDY/assets/105060483/9c53fa98-7d85-4c49-9a24-4c11cf699764)

## Dataset and preprocessing

### Download the public dataset
* [UCI Message](http://konect.cc/networks/opsahl-ucsocial)
  
* [Digg](http://konect.cc/networks/munmun_digg_reply)
  
* [Bitcoin-OTC](http://snap.stanford.edu/data/soc-sign-bitcoin-otc)

* [Bitcoin-Alpha](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha)
 
* [ia-contacts-dublin](https://networkrepository.com/ia-contacts-dublin.php)

* [fb-forum](https://networkrepository.com/fb-forum.php)

## Usage
###  Training the CADDY Dynamic graph neural network
```
python train.py --data uci  --n_epoch 20  --lr 0.1  --hidden_size 32  --alpha_ 0.5  --labda_reg 0.0001 --node_dim 16  --edge_agg mean  --ratio 0.3  --anomaly_ratio 0.1
```

## Requirements
* python >= 3.9

* Dependency

```{bash}
torch==1.12.1
networkx==2.7.1
tqdm==4.64.0
numpy==1.24.3
scikit-learn==1.0.2
```

## Command and configurations
### General flags
