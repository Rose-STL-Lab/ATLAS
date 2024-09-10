import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class LieGenerator(nn.Module):
    def __init__(self, n_channel, group_action, basis):
        super(LieGenerator, self).__init__()
        self.n_channel = n_channel
        self.sigma = nn.Parameter(torch.eye(n_channel, n_channel))
        self.mu = nn.Parameter(torch.zeros(n_channel))
        self.l0reg = False

        self.basis = basis
        self.group_action = group_action

        self.Li = nn.Parameter(torch.randn(n_channel, len(basis)))
        torch.nn.init.normal_(self.Li, 0, 0.1)

    def normalize_factor(self):
        trace = torch.einsum('kd,kd->k', self.Li, self.Li)
        factor = torch.sqrt(trace / self.Li.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self):
        return self.Li / (self.normalize_factor() + 1e-6)

    def channel_corr(self, killing=False):
        Li = self.normalize_L()
        return torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', Li, Li), diagonal=1)))

    def forward(self, x, y):  # random transformation on x
        batch_size = x.shape[0]
        z = self.sample_coefficient(batch_size, x.device)
        merged = torch.sum(self.Li.unsqueeze(-1).unsqueeze(-1) * self.basis, dim=-3)
        g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, merged))

        x_t = self.group_action(g_z, x, False)
        y_t = self.group_action(g_z, y, True)
        return x_t, y_t

    def sample_coefficient(self, batch_size, device):
        return torch.randn(batch_size, self.n_channel, device=device) # @ self.sigma + self.mu
    
    def getLi(self):
        return self.Li


class LieDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(LieDiscriminator, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
        xy = torch.cat((x, y), dim=1)
        validity = self.model(xy)
        return validity

class LieDiscriminatorSegmentation(nn.Module):
    def __init__(self, input_channels, mlp_input_size, n_class, emb_size=32):
        super(LieDiscriminatorSegmentation, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels + emb_size, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.emb = nn.Embedding(n_class, emb_size)

    def forward(self, x, y):
        _, y_ind = torch.max(y, dim=-3)
        y_emb = self.emb(y_ind).swapaxes(-1, -2).swapaxes(-2, -3)
        xy = torch.cat((x, y_emb), dim=-3)
        raw = self.model(xy)
        raw = torch.flatten(raw, -3)
        validity = self.mlp(raw)
        return validity


class LieDiscriminatorEmb(nn.Module):
    def __init__(self, input_size, n_class=2, emb_size=32):
        super(LieDiscriminatorEmb, self).__init__()
        self.input_size = input_size
        self.n_class = n_class
        self.emb_size = emb_size
        self.model = nn.Sequential(
            nn.Linear(input_size + emb_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.emb = nn.Embedding(n_class, emb_size)

    def forward(self, x, y):
        x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
        y = self.emb(y).squeeze(1)
        xy = torch.cat((x, y), dim=1)
        validity = self.model(xy)
        return validity
