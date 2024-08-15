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

        # For climate experiment
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--pred_batch_size', type=int, default=8)
        parser.add_argument('--field_length', type=int, default=4)
        parser.add_argument('--label_length', type=int, default=3)
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
        self.lr = args.lr
        self.seed = args.seed
        self.pred_batch_size = args.pred_batch_size
        self.field_length = args.field_length
        self.label_length = args.label_length
        self.fields = {"TMQ": {"mean": 19.21859, "std": 15.81723}, 
                        "U850": {"mean": 1.55302, "std": 8.29764},
                        "V850": {"mean": 0.25413, "std": 6.23163},
                        "PSL": {"mean": 100814.414, "std": 1461.2227} 
                        }
        self.device = get_device()

