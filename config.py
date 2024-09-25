import argparse
import torch
import numpy
import random
from utils import get_device


class Config:
    def __init__(self, n=10000, epochs=30, bs=16):
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--batch_size', type=int, default=bs, help='batch size')
        parser.add_argument('--epochs', type=int, default=epochs)
        parser.add_argument('--N', type=int, default=n,
                            help='for randomly generated datasets, the number of elements to generate')

        parser.add_argument('--reuse_predictor', default=False, action='store_true')
        parser.add_argument('--fixed_seed', default=False, action='store_true')
        parser.add_argument('--task', type=str)
        args = parser.parse_args()

        if args.fixed_seed:
            torch.manual_seed(0)
            numpy.random.seed(0)
            random.seed(0)

        self.debug = args.debug
        self.standard_basis = args.standard_basis
        self.batch_size = args.batch_size
        self.N = args.N
        self.epochs = args.epochs
        self.device = get_device()

        self.reuse_predictor = args.reuse_predictor
        self.task = args.task

