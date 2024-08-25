import sys
import argparse
from utils import get_device

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--bs', type=int, default=16, help='batch size')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--N', type=int, default=10000, help='for randomly generated datasets, the number of elements to generate')
        parser.add_argument('--num_pops', type=int, default=12, help='number of populations to use in genetic algorithm')
        parser.add_argument('--pop_size', type=int, default=100, help='size of each population in the genetic algorithm')
        parser.add_argument('--reuse_predictor', default=False, action='store_true')
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.batch_size = args.bs
        self.N = args.N
        self.epochs = args.epochs
        self.reuse_predictor = args.reuse_predictor
        self.device = get_device()

        self.num_pops = args.num_pops
        self.pop_size = args.pop_size

