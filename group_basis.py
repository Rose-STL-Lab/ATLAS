import torch
import torch.nn as nn
import sys
import numpy as np

from utils import get_device, rmse
device = get_device()

class GroupBasis(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_dim, transformer, num_basis, num_cosets, lr=5e-4, coeff_epsilon=1e-1, dtype=torch.float32):
=======
    def __init__(self, input_dim, transformer, num_basis, standard_basis, loss_type='rmse', lr=5e-4, reg_fac=0.05, invar_fac=3, coeff_epsilon=1e-1, dtype=torch.float32):
>>>>>>> generalized_transform
        super().__init__()
    
        self.transformer = transformer
<<<<<<< HEAD
        self.coeff_epsilon = coeff_epsilon
        self.num_basis = num_basis
=======
        self.input_dim = input_dim
        self.num_basis = num_basis
        self.coeff_epsilon = coeff_epsilon
        self.dtype = dtype
        self.invar_fac = invar_fac
        self.reg_fac = reg_fac
        self.loss_type = loss_type
        self.standard_basis = standard_basis
>>>>>>> generalized_transform

        self.lie_basis = nn.Parameter(torch.empty((num_basis, input_dim, input_dim), dtype=dtype).to(device))
        nn.init.normal_(self.lie_basis, 0, 0.02)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def summary(self):
        return self.normalize_basis(self.lie_basis).data 

    def normalize_basis(self, tensor):
        trace = torch.abs(torch.einsum('kdf,kdf->k', tensor, tensor))
        factor = torch.sqrt(trace / tensor.shape[1]) + 1e-6
        return tensor / factor.unsqueeze(-1).unsqueeze(-1)

<<<<<<< HEAD
    def similarity_loss(self, e, x):
=======
    def similarity_loss(self, x):
>>>>>>> generalized_transform
        if len(x) <= 1:
            return 0

        def derangement(n):
<<<<<<< HEAD
            """
=======
            """ 
>>>>>>> generalized_transform
                generates random derangement (not uniformly, just cycles)
            """

            perm = torch.tensor(list(range(n))).to(device)
            return torch.roll(perm, (1 + np.random.randint(n - 1)))

        xp = x[derangement(len(x))]
<<<<<<< HEAD
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
=======
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
>>>>>>> generalized_transform


    def sample_coefficients(self, bs):
        """
            Important, even when we are dealing with complex values,
            our goal is still only to find the real Lie groups so that the sampled coefficients are
            to be taken only as real numbers.
        """
        num_key_points = self.transformer.num_key_points()
<<<<<<< HEAD
        return torch.normal(0, self.coeff_epsilon, (bs, num_key_points, self.num_basis)).to(device)
=======
        return torch.normal(0, self.coeff_epsilon, (bs, num_key_points, self.num_basis)).to(device) 
>>>>>>> generalized_transform

    def apply(self, x):
        """
            x is a batched tensor with dimension [bs, *manifold_dims, \sum input_dims]
            For instance, if the manifold is 2d and each vector on the feature field is 3d
            x would look like [bs, manifold_x, manifold_y, vector_index]
        """
        bs = x.shape[0]

        coeffs = self.sample_coefficients(bs)
        
        norm = self.normalize_basis(self.lie_basis) 
        sampled = torch.sum(norm * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)

        ret = self.transformer.apply_lie(sampled, x)

        return ret

    def loss(self, xx, yy):
        if self.loss_type == 'rmse':
            raw = rmse(xx, yy)
        else:
            raw = nn.functional.cross_entropy(xx, yy)

<<<<<<< HEAD
    # called by LocalTrainer during training
    def loss(self, ypred, ytrue):
        return nn.functional.cross_entropy(ypred, ytrue)
        # return torch.sqrt(torch.mean(torch.square(ytrue - ypred)) + 1e-6)

    def regularization(self, e):
        # regularization:
        # aim for as 'orthogonal' as possible basis matrices
        r1 = self.similarity_loss(e, self.normalized_continuous())
=======
        return raw * self.invar_fac

    # called by LocalTrainer during training
    def regularization(self, e):
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = self.similarity_loss(self.lie_basis)
>>>>>>> generalized_transform

        return r1 * self.reg_fac
