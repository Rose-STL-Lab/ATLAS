import sys
import torch
from torch import nn
import pandas as pd
import numpy as np
import tqdm
from ff import R2FeatureField
from utils import get_device
from local_symmetry import Predictor
from config import Config

device = get_device()

IN_RAD = 14
OUT_RAD = 14
CLASS_WEIGHTS = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device)
NUM_CLASS = len(CLASS_WEIGHTS)


class PDEFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data)

        w = self.data.shape[-1]
        h = self.data.shape[-2]
        locs = [(h / 6, w * 0.5), (h * 0.5, w * 0.5), (h * 5 / 6, w * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]
    
    def regions(self, radius):
        return 0


class SinglePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        ).to(device)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, NUM_CLASS, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(NUM_CLASS),

# rotationally and scaling symmetric
def pde():
    pass

class PDEFeatureField():
    pass

class PDESinglePredictor():
    pass

class PDEDataset():
    pass

def discover(c, algebra, cosets):
    g = GroupBasis()
    pass

if __name__ == '__main__':
    c = Config()

    if c.task == 'discover':
        discover(c, True, True)
    elif c.task == 'discover_algebra':
        discover(c, True, False)
    elif c.task == 'discover_cosets':
        discover(c, False, True)
    else:
        print("Unknown task for PDE")
