import sys
import argparse
from utils import get_device

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--bs', type=int, action=64)
        parser.add_argument('--epochs', type=int, action=30)
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.batch_size = parsers.bs
        self.epochs = parser.epochs
        self.device = get_device()

