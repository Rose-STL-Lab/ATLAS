import torch
import torch.nn as nn
import numpy as np

import config
from config import *
from utils import mae, get_device, c64
device = get_device()


class GroupBasis(nn.Module):
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
            nn.init.normal_(tensor, 0, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def input_dimension(self):
        return self.input_dim

    def similarity_loss(self, x):
        def derangement(n):
            """ 
                generates random derangement (not necessarily uniform)
            """

            perm = torch.tensor(list(range(n))).to(device)
            for i in range(1, n):
                j = np.random.randint(i)
                perm[[i, j]] = perm[[j, i]]

            return perm

        xp = x[derangement(len(x))]
        return torch.abs(torch.sum(x * xp)) / torch.sqrt(torch.sum(x * x + xp * xp))

    # From LieGan
    def normalize_factor(self):
        trace = torch.einsum('kdf,kdf->k', self.continuous, self.continuous)
        factor = torch.sqrt(trace / self.continuous.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalized_continuous(self):
        return self.continuous / self.normalize_factor()

    def sample_coefficients(self, bs):
        num_key_points = self.transformer.num_key_points()
        unnormalized = torch.abs(torch.normal(0, 1, (bs, num_key_points, self.continuous.shape[0])).to(device))
        return unnormalized / torch.sum(unnormalized, dim=1, keepdim=True)

    def apply(self, x):
        """
            x is a batched tensor with dimension [bs, *manifold_dims, *input_dims]
            For instance, if the manifold is 2d and each vector on the feature field is 3d
            x would look like [bs, manifold_x, manifold_y, vector_index]
        """
        bs = x.shape[0]

        # `continuous` is still in the lie algebra, the feature field is responsible for doing the matrix
        # exp call
        coeffs = self.sample_coefficients(bs)
        continuous = torch.sum(self.normalized_continuous() * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)

        # conceptually, selecting which component of the lie group to use for each member of the batch
        discrete = self.discrete[torch.randint(self.discrete.shape[0], (bs, )).to(device)]

        if not config.ONLY_IDENTITY_COMPONENT:
            # train either discrete or continuous in one round
            if np.random.random() > 1:
                discrete = None
            else:
                continuous = None

        # conceptually, this is g * x. The feature field object defines how exactly to apply g
        return self.transformer.apply(discrete, continuous, x)

    # called by LocalTrainer during training
    def loss(self, ypred, ytrue):
        return torch.sqrt(torch.mean(torch.square(ytrue - ypred)) + 1e-6)

    def regularization(self):
        # regularization:
        # aim for as 'orthogonal' as possible basis matrices
        r1 = (self.similarity_loss(self.discrete) +
              self.similarity_loss(self.continuous))

        return r1 * IDENTITY_COLLAPSE_REGULARIZATION
