import torch
import torch.nn as nn
import sys
import numpy as np

from utils import get_device, rmse, transform_atlas
device = get_device()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class GroupBasis(nn.Module):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, 
            loss_type='rmse', in_rad=10, out_rad=5, lr=5e-4, reg_fac=1, coeff_epsilon=0.3,
            identity_in_rep=False, identity_out_rep=False, in_interpolation='bilinear', out_interpolation='bilinear', dtype=torch.float32,
        ):
        super().__init__()
    
        self.in_dim = in_dim
        self.man_dim = man_dim
        self.out_dim = out_dim

        self.in_rad = in_rad
        self.out_rad = out_rad

        self.in_interpolation = in_interpolation
        self.out_interpolation = out_interpolation

        self.identity_in_rep = identity_in_rep
        self.identity_out_rep = identity_out_rep

        self.num_basis = num_basis
        self.dtype = dtype
        self.reg_fac = reg_fac
        self.loss_type = loss_type
        self.standard_basis = standard_basis
        self.coeff_epsilon = coeff_epsilon

        self.lie_basis = nn.Parameter(torch.empty((num_basis, man_dim, man_dim), dtype=dtype).to(device))
        self.in_basis = nn.Parameter(torch.empty((num_basis, in_dim, in_dim), dtype=dtype).to(device))
        self.out_basis = nn.Parameter(torch.empty((num_basis, out_dim, out_dim), dtype=dtype).to(device))

        for tensor in [self.lie_basis, self.in_basis, self.out_basis]:
            nn.init.normal_(tensor, 0, 0.02)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def summary(self):
        # so normalize of the lie basis may seem as if the three basis 
        # are 'out of scale', but this is what's done for loss as well so it's okay
        ret = []
        if not self.identity_in_rep:
            ret.append(self.in_basis.data)
        ret.append(self.normalize(self.lie_basis.data))
        if not self.identity_out_rep:
            ret.append(self.in_basis.data)

        return ret


    def normalize(self, x):
        # from lie gan
        trace = torch.einsum('kdf,kdf->k', x, x)
        factor = torch.sqrt(trace / x.shape[1])
        x = x / factor.unsqueeze(-1).unsqueeze(-1)
        return x

    def similarity_loss(self, x):
        if len(x) <= 1:
            return 0
        
        x = self.normalize(x)
        if self.standard_basis:
            x = torch.abs(x)

        return torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', x, x), diagonal=1)))


    def sample_coefficients(self, bs):
        """
            Important, even when we are dealing with complex values,
            our goal is still only to find the real Lie groups so that the sampled coefficients are
            to be taken only as real numbers.
        """
        return torch.normal(0, self.coeff_epsilon, (*bs, self.num_basis)).to(device) 

    def step(self, x, y, pred):
        bs = x.batch_size()

        coeffs = self.sample_coefficients((bs, x.num_charts()))

        def sample(raw):
            return torch.matrix_exp(torch.sum(raw * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3))
       
        sampled_lie = sample(self.normalize(self.lie_basis))
        sampled_in =  sample(self.in_basis) 
        sampled_out = sample(self.out_basis)

        if self.identity_in_rep:
            sampled_in = torch.eye(self.in_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        if self.identity_out_rep:
            sampled_out = torch.eye(self.out_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        x_atlas = x.regions(self.in_rad)
        g_x_atlas = transform_atlas(sampled_lie, sampled_in, x_atlas, self.in_interpolation)

        y_atlas = y.regions(self.out_rad)
        g_y_atlas = transform_atlas(sampled_lie, sampled_out, y_atlas, self.out_interpolation)

        y_atlas_true = pred.run(g_x_atlas)

        if self.loss_type == 'rmse':
            raw = rmse(g_y_atlas, y_atlas_true)
        elif self.loss_type == 'cross_entropy':
            g_y_atlas = g_y_atlas.permute(0, 1, 3, 4, 2).flatten(0, 3)
            y_atlas_true = y_atlas_true.permute(0, 1, 3, 4, 2).flatten(0, 3)
            raw = torch.nn.functional.cross_entropy(y_atlas_true, g_y_atlas)
        else:
            raise ValueError()

        return raw

    # called by LocalTrainer during training
    def regularization(self, e):
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = self.similarity_loss(self.lie_basis)

        return r1 * self.reg_fac
