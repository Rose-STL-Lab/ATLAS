import sys
import numpy as np
import torch
import torch.nn as nn
import random
from copy import deepcopy
import scipy.linalg as SL
import math

# symmetries
from lie_gg.models import MLP
from lie_gg.liegg import polarization_matrix_2, symmetry_metrics
from lie_gg.datasets import RotoMNIST
from lie_gg.utils import L2_normed_net, count_parameters

import matplotlib.pyplot as plt

import tqdm

from config import Config
from experiment_pde import *
from utils import get_device, rmse
device = get_device()

# number of data points for polarization matrix
N = 100

sampling_range = range(16, 128 - 17)

class GlobalPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 1, kernel_size=3, padding='same'),
        ).to(device)


    def forward(self, x):
        return self.model(x)

def polarization_matrix_2(model, data):
    # LieGG implementation with the groups acting on R^2
    # data: torch.FloatTensor(B, 28, 28)
    data = data.squeeze(1)

    B, H, W = data.shape
    assert H == W

    # compute image grads
    data_grad_x = data[:, 1:, :-1] - data[:, :-1, :-1]
    data_grad_y = data[:, :-1, 1:] - data[:, :-1, :-1]
    dI = torch.stack([data_grad_x, data_grad_y], -1)

    _,h,w,_ = dI.shape

    # compute value for every single output pixel
    # maybe? can be optimized

    c_stack = []
    for x in sampling_range:
        for y in sampling_range:
            data.grad = None
            data.requires_grad = True
            data.retain_grad()

            output = model(data.view(B, 1, H, W))

            # change in output with respect to transformation
            out_grad_x = output[:, 0, x + 1, y] - output[:, 0, x, y]
            out_grad_y = output[:, 0, x, y + 1] - output[:, 0, x, y]
            xn = x / (H // 2) - 1
            yn = y / (H // 2) - 1
            OC = torch.stack([
                torch.stack([out_grad_x * xn, out_grad_x * yn]),
                torch.stack([out_grad_y * xn, out_grad_y * yn]),
            ]).permute(2, 0, 1)

            # only focus on this pixel
            output = output[:, 0, x, y]

            output.backward(torch.ones_like(output))

            dF = data.grad[:, :-1, :-1]

            # coordinate mask
            xy = torch.meshgrid(torch.arange(0, h), torch.arange(0, h), indexing='ij')
            xy = torch.stack(xy, -1).to(dI.device)
            xy = xy / (H // 2) - 1

            # collect into the network polarization matrix
            C = dF[..., None, None] * dI[..., None] * xy[None, :, :, None, :]
            C = C.view(B, -1, 2, 2).sum(1)
            
            c_stack.append((C - OC).data)

    ret = torch.cat(c_stack).reshape(B * (len(sampling_range) ** 2), 4)
    return ret

if __name__ == '__main__':
    config = Config()

    # we are just using ground truth function 
    """
    if config.reuse_predictor:
        net = torch.load('predictors/liegg_pde.pt')
    else:
        net = GlobalPredictor()
        
        train_dataset = PDEDataset(config.N, use_boundary=True)
        optim = torch.optim.Adam(net.parameters())
        dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)

        for e in range(config.epochs):
            lmean = []
            for xx, yy in tqdm.tqdm(dl):
                loss = (yy - net(xx)).abs().mean()
                optim.zero_grad()
                loss.backward()
                optim.step()

                lmean.append(float(loss.cpu().detach()))

            print("Loss", np.mean(lmean))

        torch.save(net, "predictors/liegg_pde.pt")
    """

    dataset = PDEDataset(N, use_boundary=True)
    def net(x):
        rmax = 128
        r = torch.arange(rmax, device=device)
        u, v = torch.meshgrid(r, r, indexing='ij')
        u = u.unsqueeze(0).tile(len(x), 1, 1).float() / rmax
        v = v.unsqueeze(0).tile(len(x), 1, 1).float() / rmax
        boundary = (EXCLUSION_X[0] < u) & (u < EXCLUSION_X[1]) & (EXCLUSION_Y[0] < v) & (v < EXCLUSION_Y[1])
        boundary[:] = False
        return heat_pde(x, boundary.unsqueeze(1), math.sqrt(2))
        
    # compute the network polarization matrix
    E = polarization_matrix_2(net, dataset.X)

    singular_values, symmetry_biases, generators = symmetry_metrics(E)

    torch.set_printoptions(sci_mode=False)
    for i in range(len(generators)):
        print("Generator:\n", generators[i].cpu().data)
        print("Group Element:\n", torch.matrix_exp(generators[i].cpu().data))
        print("Singular Value:", float(singular_values[i].cpu().data))
        print()
