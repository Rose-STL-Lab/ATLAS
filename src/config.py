import sys
import argparse
from utils import get_device

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.device = get_device()

