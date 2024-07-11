import torch
import torch.nn as nn
import numpy as np

import config
from utils import get_device
device = get_device()


class GroupBasis(nn.Module):
    def __init__(self, input_dim, transformer, num_basis, num_cosets, lr=5e-4, coeff_epsilon=1e-1, dtype=torch.float32):
        super().__init__()
      
        self.input_dim = input_dim
        self.transformer = transformer
        self.coeff_epsilon = coeff_epsilon
        self.num_basis = num_basis

        # lie elements
        self.continuous = nn.Parameter(torch.empty((num_basis, input_dim, input_dim), dtype=dtype).to(device))
        # normal matrices
        self.discrete = nn.Parameter(torch.empty((num_cosets, input_dim, input_dim), dtype=dtype).to(device))

        for tensor in [
            self.discrete,
            self.continuous,
        ]:
            nn.init.normal_(tensor, 0, 0.02)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def input_dimension(self):
        return self.input_dim

    def similarity_loss(self, e, x):
        if len(x) <= 1:
            return 0

        def derangement(n):
            """
                generates random derangement (not uniformly, just cycles)
            """

            perm = torch.tensor(list(range(n))).to(device)
            return torch.roll(perm, (1 + np.random.randint(n - 1)))

        xp = x[derangement(len(x))]
        denom = torch.sum(torch.real(x * torch.conj(x)), dim=(-2, -1))

        """
        if self.dtype == torch.complex64:
            return torch.abs(torch.sum((
                torch.real(x) * torch.real(xp) +
                torch.imag(x) * torch.imag(xp)
            ) / denom.unsqueeze(-1).unsqueeze(-1)))
        else:
            return torch.abs(torch.sum(x * xp / denom))
        """

        # constrained
        return torch.sum(torch.abs(x * xp / denom.unsqueeze(-1).unsqueeze(-1)))

    # From LieGan
    def normalize_factor(self):
        trace = torch.abs(torch.einsum('kdf,kdf->k', self.continuous, self.continuous))
        factor = torch.sqrt(trace / self.continuous.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalized_continuous(self):
        return self.continuous / self.normalize_factor()

    def sample_coefficients(self, bs):
        """
            Important, even when we are dealing with complex values,
            our goal is still only to find the real Lie groups so that the sampled coefficients are
            to be taken only as real numbers.
        """
        num_key_points = self.transformer.num_key_points()
        return torch.normal(0, self.coeff_epsilon, (bs, num_key_points, self.num_basis)).to(device)

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
            if np.random.random() > 0.5:
                discrete = None
            else:
                continuous = None

        # conceptually, this is g * x. The feature field object defines how exactly to apply g
        return self.transformer.apply(discrete, continuous, x)

    # called by LocalTrainer during training
    def loss(self, ypred, ytrue):
        return nn.functional.cross_entropy(ypred, ytrue)
        # return torch.sqrt(torch.mean(torch.square(ytrue - ypred)) + 1e-6)

    def regularization(self, e):
        # regularization:
        # aim for as 'orthogonal' as possible basis matrices
        r1 = self.similarity_loss(e, self.normalized_continuous())

        return r1 * config.IDENTITY_COLLAPSE_REGULARIZATION
