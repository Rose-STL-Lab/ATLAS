import torch
import torch.nn as nn
import numpy as np

from config import *
from local_symmetry import Basis
from utils import mae, get_device, c64
device = get_device()

class GroupBasis(Basis, nn.Module):
    def __init__(self, input_dim, transformer, num_mats, lr=5e-4):
        super().__init__()
      
        self.input_dim = input_dim
        self.transformer = transformer

        self.continuous = nn.Parameter(torch.empty((num_mats, input_dim, input_dim)).to(device))
        self.discrete = nn.Parameter(torch.empty((num_mats, input_dim, input_dim)).to(device))

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
        return self.loss(x, x[self.derangement()])

    def epsilon(self):
        return 0.1

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
        r1 += self.similarity_loss(torch.det(self.discrete))
        r1 += self.similarity_loss(torch.det(self.continuous))

        num_key_points = self.transformer.num_key_points()
        identity = torch.eye(self.input_dimension()).to(device)

        mixing_factor = torch.random.rand((bs, num_key_points, self.continuous.shape[0])).to(device)
        mixing_factor /= torch.sum(mixing_factor, dim=-1) # now a probability distribution
        continuous = self.epsilon() * (self.continuous * mixing_factor.unsqueeze(-1).unsqueeze(-1) - identity)
        discrete = self.discrete[torch.random.randint(self.discrete.shape[0], (bs, )).to(device)]
        key_points = discrete * continuous

        xp = self.transformer.apply(key_points, x)

        return xp, r1 * IDENTITY_COLLAPSE_REGULARIZATION 

