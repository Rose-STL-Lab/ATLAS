import torch
import torch.nn as nn
import sys
import numpy as np

from utils import get_device, rmse
device = get_device()

class Homomorphism:
    def apply_y(self, x_action, y):
        y_action = self.forward(x_action)
        return (y_action @ y.unsqueeze(-1)).squeeze(-1)

class TrivialHomomorphism(Homomorphism):
    def __init__(self, manifold_shape, ff_dim):
        self.identity = torch.eye(ff_dim, ff_dim).expand(*manifold_shape, ff_dim, ff_dim)

    def forward(self, x):
        return self.identity

    def apply_y(self, x_action, y):
        return y

class GroupBasis(nn.Module):
    def __init__(self, input_dim, transformer, homomorphism, num_basis, standard_basis, loss_type='rmse', lr=5e-4, reg_fac=0.05, invar_fac=3, coeff_epsilon=1e-1, dtype=torch.float32):
        super().__init__()

        self.input_dim = input_dim
        self.num_basis = num_basis
        self.coeff_epsilon = coeff_epsilon
        self.dtype = dtype
        self.invar_fac = invar_fac
        self.reg_fac = reg_fac
        self.loss_type = loss_type
        self.standard_basis = standard_basis

        self.transformer = transformer
        self.homomorphism = homomorphism
        self.lie_basis = nn.Parameter(torch.empty((num_basis, input_dim, input_dim), dtype=dtype).to(device))
        nn.init.normal_(self.lie_basis, 0, 0.02)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def summary(self):
        return self.normalize_basis(self.lie_basis).data 

    def normalize_basis(self, tensor):
        trace = torch.abs(torch.einsum('kdf,kdf->k', tensor, tensor))
        factor = torch.sqrt(trace / tensor.shape[1]) + 1e-6
        return tensor / factor.unsqueeze(-1).unsqueeze(-1)

    def similarity_loss(self, x):
        if len(x) <= 1:
            return 0

        def derangement(n):
            """ 
                generates random derangement (not uniformly, just cycles)
            """

            perm = torch.tensor(list(range(n))).to(device)
            return torch.roll(perm, (1 + np.random.randint(n - 1)))

        xp = x[derangement(len(x))]
        denom = torch.sum(torch.real(x * torch.conj(x)), dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        if self.standard_basis:
            if self.dtype == torch.complex64:
                return torch.sum((
                    torch.abs(torch.real(x) * torch.real(xp)) + 
                    torch.abs(torch.imag(x) * torch.imag(xp))
                ) / denom)
            else:
                return torch.sum(torch.abs(x * xp / denom))
        else:
            if self.dtype == torch.complex64:
                return torch.abs(torch.sum((
                    torch.real(x) * torch.real(xp) + 
                    torch.imag(x) * torch.imag(xp)
                ) / denom))
            else:
                return torch.abs(torch.sum(x * xp / denom))


    def sample_coefficients(self, bs):
        """
            Important, even when we are dealing with complex values,
            our goal is still only to find the real Lie groups so that the sampled coefficients are
            to be taken only as real numbers.
        """
        num_key_points = self.transformer.num_key_points()
        return torch.normal(0, self.coeff_epsilon, (bs, num_key_points, self.num_basis)).to(device) 

    def apply(self, x, y):
        """
            x is a batched tensor with dimension [bs, *manifold_dims, \sum input_dims]
            For instance, if the manifold is 2d and each vector on the feature field is 3d
            x would look like [bs, manifold_x, manifold_y, vector_index]
        """
        bs = x.shape[0]

        coeffs = self.sample_coefficients(bs)
        
        norm = self.normalize_basis(self.lie_basis) 
        sampled = torch.sum(norm * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)

        full_lie = self.transformer.smooth_function(sampled)
        full_exp = torch.matrix_exp(full_lie)
        ret = (full_exp @ x.unsqueeze(-1)).squeeze(-1)

        return ret, self.homomorphism.apply_y(full_exp, y)

    def loss(self, xx, yy):
        if self.loss_type == 'rmse':
            raw = rmse(xx, yy)
        else:
            raw = nn.functional.cross_entropy(xx, yy)

        return raw * self.invar_fac

    # called by LocalTrainer during training
    def regularization(self, e):
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = self.similarity_loss(self.lie_basis)

        return r1 * self.reg_fac
