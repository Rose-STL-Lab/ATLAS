import torch
import torch.nn as nn
import numpy as np

from utils import get_device, rmse
device = get_device()

class FFConfig:
    # kind = 'lie' or 'jacobian' 
    # if kind = 'jacobian', then lambda_dim is required
    # vs_dim = vector space dimension of the feature field
    def __init__(self, kind, vs_dim, manifold_dim=None, lambda_dim=None):
        if kind == 'jacobian':
            assert lambda_dim is not None
        else:
            assert lambda_dim is None

        self.kind = kind
        self.vs_dim = vs_dim
        self.lambda_dim = lambda_dim
        self.manifold_dim = manifold_dim

class FeatureFieldBasis(nn.Module):
    def __init__(self, ff_config, num_basis, dtype):
        super().__init__()
        self.config = ff_config
        self.dtype = dtype

        if ff_config.kind == 'lie':
            self.tensor = nn.Parameter(torch.empty((num_basis, ff_config.vs_dim, ff_config.vs_dim), dtype=dtype).to(device))
            nn.init.normal_(self.tensor, 0, 0.02)
        else:
            self.tensor = nn.Parameter(torch.empty((num_basis, ff_config.lambda_dim), dtype=dtype).to(device))
            self.b = nn.Parameter(torch.empty((ff_config.vs_dim, ff_config.lambda_dim * ff_config.manifold_dim)).to(device))

            nn.init.normal_(self.tensor, 0, 0.02)
            nn.init.normal_(self.b, 0, 0.02)

    def similarity_loss(self, e, x):
        def derangement(n):
            """ 
                generates random derangement (not uniformly, just cycles)
            """

            perm = torch.tensor(list(range(n))).to(device)
            return torch.roll(perm, (np.random.randint(n)))

        xp = x[derangement(len(x))]
        denom = torch.sum(torch.real(x * torch.conj(x)))

        if self.dtype == torch.complex64:
            if e < 5:
                return torch.abs(torch.sum(
                    torch.real(x) * torch.real(xp) + 
                    torch.imag(x) * torch.imag(xp)
                )) / denom

            return torch.sum(
                torch.abs(torch.real(x) * torch.real(xp)) + 
                torch.abs(torch.imag(x) * torch.imag(xp))
            ) / denom
        else:
            if e < 5:
                return torch.abs(torch.sum(
                    x * xp 
                )) / denom

            return torch.sum(torch.abs(x * xp)) / denom

    def reg(self, e):
        if self.config.kind == 'lie':
            return self.similarity_loss(e, self.tensor)
        else:
            # encourage non trivial B matrix with many values, but only to a certain point (min 1)
            return self.similarity_loss(e, self.tensor) -  \
                torch.sum(torch.minimum(torch.tensor(1).to(device), torch.abs(self.b)))

class GroupBasis(nn.Module):
    def __init__(self, input_ffs, transformer, num_basis, invar_fac=3, reg_fac=0.1, lr=3e-4, coeff_epsilon=1e-1, dtype=torch.float32):
        super().__init__()
        self.transformer = transformer
        self.num_basis = num_basis
        self.coeff_epsilon = coeff_epsilon
        self.dtype = dtype
        self.invar_fac = invar_fac
        self.reg_fac = reg_fac

        self.inputs = nn.ModuleList([FeatureFieldBasis(ff, num_basis, dtype) for ff in input_ffs])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def summary(self):
        return [self.normalize_basis(inp.tensor).data if inp.config.kind == 'lie' else inp.b.data for inp in self.inputs] 

    def normalize_basis(self, tensor):
        if len(tensor.shape) == 3:
            trace = torch.abs(torch.einsum('kdf,kdf->k', tensor, tensor))
            factor = torch.sqrt(trace / tensor.shape[1]) + 1e-6
            return tensor / factor.unsqueeze(-1).unsqueeze(-1)
        else:
            trace = torch.abs(torch.einsum('kd,kd->k', tensor, tensor))
            factor = torch.sqrt(trace / tensor.shape[1]) + 1e-6
            return tensor / factor.unsqueeze(-1)

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
            x is a batched tensor with dimension [bs, *manifold_dims, \sum input_dims]
            For instance, if the manifold is 2d and each vector on the feature field is 3d
            x would look like [bs, manifold_x, manifold_y, vector_index]
        """
        bs = x.shape[0]

        coeffs = self.sample_coefficients(bs)
        used = 0
        ret = torch.empty(x.shape, dtype=self.dtype).to(device)
        
        for inp in self.inputs:
            full = used + inp.config.vs_dim
            norm = self.normalize_basis(inp.tensor) 

            if inp.config.kind == 'lie':
                vector = torch.sum(norm * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)
                ret[..., used:full] = self.transformer.apply_lie(vector, x[..., used:full])
            else:
                vector = torch.sum(norm * coeffs.unsqueeze(-1), dim=-2)
                ret[..., used:full] = self.transformer.apply_jacobian(vector, inp.b, x[..., used:full])

            used = full

        return ret

    def loss(self, xx, yy):
        raw = rmse(xx, yy)
        return raw * self.invar_fac

    # called by LocalTrainer during training
    def regularization(self, e):
        # regularization:
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = 0
        for inp in self.inputs:
            r1 += inp.reg(e)

        return r1 * self.reg_fac
