import sys
import argparse
from utils import get_device

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--bs', type=int, default=64, help='batch size')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--N', type=int, default=10000, help='for randomly generated datasets, the number of elements to generate')
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.batch_size = args.bs
        self.N = args.N
        self.epochs = args.epochs
        self.device = get_device()

