import torch
import torch.nn as nn
import numpy as np

from utils import get_device
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
        denom = torch.sum(torch.real(x * torch.conj(x)))
        if self.dtype == torch.complex64:
            return torch.sum(
                torch.abs(torch.real(x) * torch.real(xp)) + 
                torch.abs(torch.imag(x) * torch.imag(xp))
            ) / denom
        else:
            return torch.sum(torch.abs(x * xp)) / denom

    def reg(self):
        if self.config.kind == 'lie':
            return self.similarity_loss(self.tensor)
        else:
            return self.similarity_loss(self.tensor) -  \
                torch.sum(torch.minimum(torch.tensor(1).to(device), torch.abs(self.b)))

class GroupBasis(nn.Module):
    # lie_epsilon: if any manifold is lie, then coefficient sampling is multiplied by that radius
    def __init__(self, input_ffs, transformer, num_basis, lr=5e-4, lie_epsilon=1e-2, dtype=torch.float32):
        super().__init__()
        self.transformer = transformer
        self.num_basis = num_basis
        self.lie_epsilon = lie_epsilon
        self.dtype = dtype

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
        raw = torch.normal(0, 1, (bs, num_key_points, self.num_basis)).to(device)

        # lie works much better with infitesimal generators
        if any(inp.config.kind == 'lie' for inp in self.inputs):
            raw *= self.lie_epsilon

        return raw

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

    # called by LocalTrainer during training
    def loss(self, ypred, ytrue):
        return torch.sqrt(torch.mean(torch.square(ytrue - ypred)) + 1e-6)

    def regularization(self):
        # regularization:
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = 0
        for inp in self.inputs:
            r1 += inp.reg()

        return r1 
