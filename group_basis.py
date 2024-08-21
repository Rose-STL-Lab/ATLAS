import torch
import torch.nn as nn
import sys
import numpy as np

from utils import get_device, rmse, transform_atlas
device = get_device()

DEBUG=0

class GroupBasis(nn.Module):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, 
            in_rad=10, out_rad=5, lr=5e-4, r1=0.1, r2=0.25, r3=2,
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
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.standard_basis = standard_basis

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
        ret.append(self.lie_basis.data)
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
        return torch.normal(0, 1, (*bs, self.num_basis)).to(device) 

    def step(self, x, pred):
        bs = x.batch_size()

        coeffs = self.sample_coefficients((bs, x.num_charts()))

        def sample(raw):
            return torch.matrix_exp(torch.sum(raw * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3))
       
        sampled_lie = sample(self.lie_basis)
        sampled_in  = sample(self.in_basis) 
        sampled_out = sample(self.out_basis)

        if self.identity_in_rep:
            sampled_in = torch.eye(self.in_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        if self.identity_out_rep:
            sampled_out = torch.eye(self.out_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        x_atlas = x.regions(self.in_rad)
        g_x_atlas = transform_atlas(sampled_lie, sampled_in, x_atlas, self.in_interpolation)

        y_atlas = pred(x_atlas)
        if pred.returns_logits():
            y_atlas = torch.nn.functional.softmax(y_atlas, dim=-3)
        y_atlas = y_atlas.detach()
        g_y_atlas = transform_atlas(sampled_lie, sampled_out, y_atlas, self.out_interpolation)

        y_atlas_true = pred.run(g_x_atlas)

        global DEBUG
        DEBUG += 1
        if DEBUG == -1:
            import matplotlib.pyplot as plt
            def plot_charts(original_charts, transformed_charts):
                bs, num_charts, ff_dim, height, width = original_charts.shape
                
                for b in range(1):
                    for n in range(1):
                        fig, axs = plt.subplots(2, ff_dim, figsize=(5*ff_dim, 10))
                        fig.suptitle(f'Batch {b+1}, Chart {n+1}')
                    
                        axs = axs.reshape(2, 1)

                        for d in range(ff_dim):
                            # Plot original chart
                            axs[0, d].imshow(original_charts[b, n, d].detach().cpu().numpy(), cmap='viridis')
                            axs[0, d].set_title(f'Original Channel {d+1}')
                            axs[0, d].axis('off')
                            
                            # Plot transformed chart
                            axs[1, d].imshow(transformed_charts[b, n, d].detach().cpu().numpy(), cmap='viridis')
                            axs[1, d].set_title(f'Transformed Channel {d+1}')
                            axs[1, d].axis('off')
                        
                        plt.tight_layout()
                        plt.show() 
            
            s = g_y_atlas.shape
            flat_y = torch.nn.functional.softmax(g_y_atlas.permute(0, 1, 3, 4, 2).flatten(0, 3)).reshape(s[0], s[1], s[3], s[4], s[2])
            flat_y = flat_y.permute(0, 1, 4, 2, 3)

            plot_charts(torch.max(flat_y, dim=-3, keepdim=True)[1], torch.max(y_atlas_true, dim=-3, keepdim=True)[1])

        raw = pred.loss(y_atlas_true, g_y_atlas)

        return raw

    # called by LocalTrainer during training
    def regularization(self, e):
        # aim for as 'orthogonal' as possible basis matrices and in general avoid identity collapse
        r1 = self.similarity_loss(self.lie_basis)

        # past a certain point, increasing the basis means nothing
        # we only want to increase to a certain extent

        clipped = self.lie_basis.clamp(-self.r2, self.r2)
        trace = torch.sqrt(torch.einsum('kdf,kdf->k', clipped, clipped))
        r2 = -torch.mean(trace)

        return self.r1 * r1 + self.r3 * r2
