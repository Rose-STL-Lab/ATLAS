import torch
import torch.nn as nn
import numpy as np

from config import *
from local_symmetry import Basis
from utils import mae, get_device, c64
device = get_device()

class GroupBasis(Basis, nn.Module):
    def __init__(self, input_dim, transformer, num_basis, num_cosets, lr=5e-4):
        super().__init__()
      
        self.input_dim = input_dim
        self.transformer = transformer

        # lie elements
        self.continuous = nn.Parameter(torch.empty((num_basis, input_dim, input_dim)).to(device))
        # normal matrices
        self.discrete = nn.Parameter(torch.empty((num_cosets, input_dim, input_dim)).to(device))

        for tensor in [
            self.discrete,
            self.continuous,
        ]:
            nn.init.normal_(tensor, 0, 2e-2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def input_dimension(self):
        return self.input_dim

    # generates random derangement (not necessarily uniform)
    def derangement(self, n):
        perm = torch.tensor(list(range(n))).to(device)
        for i in range(1, n):
            j = np.random.randint(i)
            perm[[i, j]] = perm[[j, i]]

        return perm

    def similarity_loss(self, x):
        return torch.sum(x * x[self.derangement(len(x))])


    # From LieGan
    def normalize_factor(self):
        trace = torch.einsum('kdf,kdf->k', self.continuous, self.continuous)
        factor = torch.sqrt(trace / self.continuous.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalized_continuous(self):
        return self.continuous / (self.normalize_factor() + 1e-6)

    def sample_coefficients(self, bs):
        num_key_points = self.transformer.num_key_points()
        return torch.normal(0, 1, (bs, num_key_points, self.continuous.shape[0])).to(device)

    def loss(self, ytrue, ypred):
        return torch.sqrt(torch.mean(torch.square(ytrue - ypred))) 

    def apply(self, x):
        # x assumed to have batch
        bs = x.shape[0]

        # regularization 1: 
        # aim for as 'linearly independent' as possible basis matrices 
        # (in other words, create as much of a span). 
        # since determining if one matrix can be written as a word of other 
        # matrices is computationally intensive, we use this heurestic instead
        r1 = 0
        r1 += self.similarity_loss(nn.functional.normalize(self.discrete, dim=-2))
        r1 += self.similarity_loss(nn.functional.normalize(self.continuous, dim=-2))

        # we also push for a diverse range of determinants 
        # r1 += self.similarity_loss(torch.det(self.discrete))
        # r1 += self.similarity_loss(torch.det(self.continuous))

        identity = torch.eye(self.input_dimension()).to(device)


        coeffs = self.sample_coefficients(bs)
        continuous = torch.sum(self.normalized_continuous() * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)

        discrete = self.discrete[torch.randint(self.discrete.shape[0], (bs, )).to(device)].unsqueeze(-3)

        # xp = self.transformer.apply(discrete, x)
        xp = self.transformer.apply(discrete, continuous, x)
        # xp = self.transformer.apply(key_points, x)

        return xp, r1 * IDENTITY_COLLAPSE_REGULARIZATION 

