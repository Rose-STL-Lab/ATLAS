import sys
import argparse
from utils import get_device

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--bs', type=int, default=64, help='batch size')
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--N', type=int, default=10000, help='for randomly generated datasets, the number of elements to generate')
        parser.add_argument('--n_component', type=int, default=10)
        parser.add_argument('--n_hidden', type=int, default=72)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--c_weight', type=float, default=0.005)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--log_interval', type=int, default=50)
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.batch_size = args.bs
        self.N = args.N
        self.epochs = args.epochs
        self.n_component = args.n_component
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.c_weight = args.c_weight
        self.weight_decay = args.weight_decay
        self.log_interval = args.log_interval
        self.device = get_device()

