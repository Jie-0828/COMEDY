# COMEDY: Continuous-time Anomalous Edge Detection in Dynamic Networks
<!--#### -->
## Introduction
To address anomalous edge detection in dynamic networks, in this paper, we propose a novel  <u>**C**<u>ontinuous-time An <u>**OM**<u>aly  <u>**E**<u>dge detection framework in  <u>**DY**<u>namic networks **(COMEDY)**. This framework features a specialized filter to identify and exclude outdated information from inactive nodes while preserving historical interaction information within a specific time window. Coupled with an attention mechanism and a temporal decay function, the network enhances its ability to extract coupled spatial-temporal information among nodes. Secondly, COMEDY employs tailored negative sampling strategies specifically for distinct types of real-world anomalies. These strategies enhance the model's ability to generalize across different scenarios and detect nuanced abnormalities. Moreover, considering the absence of node attributes in dynamic network datasets, we introduce a general node coding strategy suitable for continuous-time graph neural networks. A key innovation is the incorporation of node gravity centrality as a form of spatial coding. This approach provides a comprehensive representation of the nodes' connection patterns, enhancing the model’s capacity to understand and interpret complex structural dynamics within the graph.

The framework of COMEDY consists of four main components: the outdated information filter (blue box), the spatial-temporal encoding (green box), the attention-temporal aggregator (yellow box), and the anomaly detector (purple box). Below the overall architecture, index with (a). Below the Sptial-temporal node encoding (dotted green box), index with (b). Below the Attention-temporal aggregator (dotted yellow box), index with (c).
![framework](framework.png)

## Dataset and preprocessing

### Download the public dataset
* [UCI Message](http://konect.cc/networks/opsahl-ucsocial)
  
* [Digg](http://konect.cc/networks/munmun_digg_reply)
  
* [email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)

* [ia-contacts-dublin](https://networkrepository.com/ia-contacts-dublin.php)

* [sx-mathoverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)

* [sx-askubuntu](https://snap.stanford.edu/data/sx-askubuntu.html)

## Usage
###  Training the CADDY Dynamic graph neural network
```
python train.py --data uci  --n_epoch 10  --lr 0.01  --hidden_size 32  --node_dim 8  --edge_agg mean  --ratio 0.3 --dropout 0  --anomaly_ratio 0.1  --threshold 20000  --window_size 5
```

## Requirements
* python >= 3.6

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
```{txt}
optional arguments:
  -d DATA, --data DATA                       data sources to use
  --n_epoch N_EPOCH                          number of epochs
  --lr LR                                    learning rate
  --hidden_size HIDDEN_SIZE                  dimensions of the model hidden size
  --node_dim NODE_DIM                        dimensions of the node encoding
  --edge_agg {mean,had,w1,w2,activate}       Edge Aggregator(EdgeAgg) method
  --ratio                                    the ratio of training sets
  --dropout                                  dropout rate
  --anomaly_ratio                            the ratio of anomalous edges in the testing set
  --threshold                                inactive nodes threshold
  --window_size                              the queue size of the historical information bank
```
