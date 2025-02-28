import argparse

from sklearn import metrics

parser = argparse.ArgumentParser('Interface for COMEDY experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, bitcoinotc, UCI,DIGG,bitcoinalpha or Reddit', default='UCI')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_size', type=int, default=32, help='Dimentions of the node hidden size')
parser.add_argument('--node_dim', type=int, default=8, help='Dimentions of the node embedding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--ratio', type=str,help='the ratio of training sets', default=0.3)
parser.add_argument('-dropout', type=float, help='dropout', default=0)
parser.add_argument('-threshold', type=float, help='the length of active node queue', default=50)
parser.add_argument('-anomaly_radio', type=float, help='test_radio', default=0.1)
parser.add_argument('-seed', type=float, help='seed', default=1)
parser.add_argument('-sampling', type=str, help='seed', default="str")

args = parser.parse_args()
