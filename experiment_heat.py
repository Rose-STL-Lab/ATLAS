import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import get_device, ManifoldPredictor
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff import R2FeatureField
from config import Config
import torchvision
from torchvision import datasets, transforms

IN_RAD = 14
OUT_RAD = 14
NUM_CLASS = 10

device = get_device()

class HeatFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data)

        w = self.data.shape[-1]
        h = self.data.shape[-2]
        mid_c = self.data.shape[-1] // 2
        locs = [(h / 6, w * 0.5), (h * 0.5, w * 0.5), (h * 5 / 6, w * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]
    
    def regions(self, radius):
        ''' we assume a priori knowledge of a good chart
        In this case, it makes sense to do equirectangular projection (especially since that's used in the projection code),
        but theoretically similar projections should work as well'''

        max_r = self.data.shape[-2] / (2 * math.pi)
        assert radius < max_r

        ret = torch.zeros(self.data.shape[0], len(self.locs), self.data.shape[1], 2 * radius + 1, 2 * radius + 1)
        inds = torch.arange(-radius, radius + 1, device=device) / max_r
        phi = torch.asin(inds)
        phi_inds = (phi / math.pi * self.data.shape[-1] + self.data.shape[-1] // 2).round().long()

        charts = []
        for i, (p0, _) in enumerate(self.locs):
            ret[:, i] = self.data[
                :, 
                :, 
                torch.arange(p0 - radius, p0 + radius + 1, device=device).unsqueeze(0), 
                phi_inds.unsqueeze(1)
            ]

        return ret

# predicts on a single region
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
        ).to(device)

    def forward(self, x):
        enc = self.down(x)
        dec = self.up(enc)
        return dec[:, :, :29, :29]

class HeatPredictor(nn.Module, Predictor):
    def __init__(self):
        super(HeatPredictor, self).__init__()
        
        self.c1 = SinglePredictor()
        self.c2 = SinglePredictor()
        self.c3 = SinglePredictor()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.run(x)

    def run(self, x):
        return torch.stack([
            self.c1(x[:, 0]),
            self.c2(x[:, 1]),
            self.c3(x[:, 2]),
        ], dim=1)

    def loss(self, xx, yy):
        xx = xx.permute(0, 1, 3, 4, 2).flatten(0, 3)
        yy = yy.permute(0, 1, 3, 4, 2).flatten(0, 3)
        return nn.functional.cross_entropy(xx, yy)

    def name(self):
        return "heat"

    def needs_training(self):
        return True

    def returns_logits(self):
        return True


class HeatDataset(torch.utils.data.Dataset):
    def __init__(self, N, rotate=180, train=True):
        self.dataset = datasets.Heat(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

        self.w = 120
        self.h = self.w // 2

        self.x = torch.zeros(N, 1, self.w, self.h, device=device)
        self.y = torch.zeros(N, NUM_CLASS, self.w, self.h, device=device)

        h = lambda x : hash(str(x))

        for i in range(N):
            j = [h(i) % N, (h(i) + 1) % N, (h(i) + 2) % N]
            starts = [(int(self.w / 6) - 14, 0), (int(self.w / 2) - 14, 0), (int(self.w * 5 / 6) - 14, 0)]

            # x_flat/y_flat represent the digits on a cylinder
            # we then project onto the sphere (equirectangular)
            x_flat = torch.zeros(1, self.w, 28, device=device)
            y_flat = torch.zeros(NUM_CLASS, self.w, 28, device=device)

            for jp, (r, c) in zip(j, starts):
                theta = h(i + jp) % (2 * rotate) - rotate if rotate else 0
                x, y = self.dataset[jp]
                x_curr = torchvision.transforms.functional.rotate(x, theta)
                x_curr = torch.max(x_curr, torch.tensor(0.1))

                # we do a little more than 28 since otherwise the chart
                # would have a null entry at the border, and then the rotation
                # would be weird
                # this is somewhat of a hack, but is largely unimportant
                p = 2 * IN_RAD + 1
                y_curr = torch.zeros(NUM_CLASS, p, 28)

                y_curr[y] = 1

                x_flat[:, r: r + 28, c: c + 28] = x_curr
                y_flat[:, r: r + p, c: c + 28] = y_curr


            self.x[i] = self.project(x_flat)
            self.y[i] = self.project(y_flat)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def discover():
    config = Config()

    if config.reuse_predictor:
        predictor = torch.load('predictors/heat.pt')
    else:
        predictor = HeatPredictor()

    basis = GroupBasis(
        1, 2, NUM_CLASS, 1, config.standard_basis, 
        lr=5e-4, in_rad=IN_RAD, out_rad=OUT_RAD, 
        identity_in_rep=True, identity_out_rep=True, # matrix exp of 62 x 62 matrix generally becomes nan
    )

    dataset = HeatDataset(config.N, rotate=60)

    gdn = LocalTrainer(HeatFeatureField, predictor, basis, dataset, config)   
    gdn.train()

if __name__ == '__main__':
    discover()
