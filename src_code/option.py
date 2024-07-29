import argparse

parser = argparse.ArgumentParser('Interface for COMEDY experiments on anomalous edge classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, bitcoinotc, UCI,DIGG,bitcoinalpha or Reddit', default='uci')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--hidden_size', type=int, default=32, help='Dimentions of the node hidden size')
parser.add_argument('--embedding_dims', type=int, default=8, help='Dimentions of the time embedding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--divide', type=str,help='the ratio of training sets', default=0.3)
parser.add_argument('-dropout', type=float, help='dropout', default=0)
parser.add_argument('-threshold', type=float, help='temporal neighbor threshold', default=10000)
parser.add_argument('-test_radio', type=float, help='test_radio', default=0.1)
parser.add_argument('-negative_sample', type=str,choices=['time', 'spatial', 'hist'], help='negative sample strategy',default='spatial')
parser.add_argument('-weight_decay', type=float, default=5e-4)
parser.add_argument('-window_size', type=float, help='window_size', default=5)
parser.add_argument('-seed', type=float, help='seed', default=1)

args = parser.parse_args()