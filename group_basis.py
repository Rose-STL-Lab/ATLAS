import torch
import torch.nn as nn

from utils import get_device, transform_atlas
device = get_device()

DEBUG = 0

def normalize(x):
    # from lie gan
    trace = torch.einsum('kdf,kdf->k', x, x)
    factor = torch.sqrt(trace / x.shape[1])
    x = x / factor.unsqueeze(-1).unsqueeze(-1)
    return x


class GroupBasis(nn.Module):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, 
            in_rad=10, out_rad=5, lr=5e-4, r1=0.05, r2=1, r3=0.35,
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

        # self.lie_basis = nn.Parameter(torch.empty((num_basis, man_dim, man_dim), dtype=dtype).to(device))
        self.in_basis = nn.Parameter(torch.empty((num_basis, in_dim, in_dim), dtype=dtype).to(device))
        self.out_basis = nn.Parameter(torch.empty((num_basis, out_dim, out_dim), dtype=dtype).to(device))

        # for tensor in [self.in_basis, self.lie_basis, self.out_basis]:
            # nn.init.normal_(tensor, 0, 0.02)
        self.lie_basis = torch.tensor([[[1, 0], [0, 0]]])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def summary(self):
        ret = []
        if not self.identity_in_rep:
            ret.append(self.in_basis.data)
        ret.append(self.lie_basis.data)
        if not self.identity_out_rep:
            ret.append(self.in_basis.data)

        return ret

    def similarity_loss(self, x):
        if len(x) <= 1:
            return 0
        
        x = normalize(x)
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

    def step(self, x, pred, _y):
        """
            y is only used for debug
        """

        bs = x.batch_size()

        coeffs = 0 * self.sample_coefficients((bs, x.num_charts())) + 1

        def sample(raw):
            return torch.matrix_exp(torch.sum(raw * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3))
       
        sampled_lie = sample(self.lie_basis)
        sampled_in = sample(self.in_basis)
        sampled_out = sample(self.out_basis)

        if self.identity_in_rep:
            sampled_in = torch.eye(self.in_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        if self.identity_out_rep:
            sampled_out = torch.eye(self.out_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, x.num_charts(), 1, 1)

        x_atlas = x.regions(self.in_rad)
        g_x_atlas = transform_atlas(sampled_lie, sampled_in, x_atlas, self.in_interpolation)

        y_atlas = pred.run(x_atlas)
        if pred.returns_logits():
            y_atlas = torch.nn.functional.softmax(y_atlas, dim=-3)
        y_atlas = y_atlas.detach()
        g_y_atlas = transform_atlas(sampled_lie, sampled_out, y_atlas, self.out_interpolation)

        y_atlas_true = pred.run(g_x_atlas)

        global DEBUG
        DEBUG += 1
        if DEBUG >= 1:
            import matplotlib.pyplot as plt
            import math
            transforms = torch.tensor([
                [[-0.1, 0], [0, -0.1]],
                [[0, 0], [0.2, 0]],
                [[0, 0.1], [-0.1, 0]]
            ], device=device)
            
            s = lambda x : sample(x)
            x = x_atlas
            _, fx = torch.max(pred.run(x_atlas), dim=-3)

            g1x = transform_atlas(s(transforms[0]), sampled_in, x_atlas, 'bilinear')
            _, fg1x = torch.max(pred.run(g1x), dim=-3)
            _, g1fx = torch.max(transform_atlas(s(transforms[0]), sampled_out, y_atlas, 'bilinear'), dim=-3)
            g1 = [g1x, fg1x, g1fx, "scale"]

            g2x = transform_atlas(s(transforms[1]), sampled_in, x_atlas, 'bilinear')
            _, fg2x = torch.max(pred.run(g2x), dim=-3)
            _, g2fx = torch.max(transform_atlas(s(transforms[1]), sampled_out, y_atlas, 'bilinear'), dim=-3)
            g2 = [g2x, fg2x, g2fx, "shear"]

            g3x = transform_atlas(s(transforms[2]), sampled_in, x_atlas, 'bilinear')
            _, fg3x = torch.max(pred.run(g3x), dim=-3)
            _, g3fx = torch.max(transform_atlas(s(transforms[2]), sampled_out, y_atlas, 'bilinear'), dim=-3)
            g3 = [g3x, fg3x, g3fx, "rot"]

            fig, axs = plt.subplots(3, 4, figsize=(10, 5))

            axs = axs.reshape(3, 4)

            p = 175
            def r(x):
                return x[x.shape[0] // 2 - p: x.shape[0] // 2 + p, x.shape[1] // 2 - p: x.shape[1] // 2 + p]

            low = x[0, 0, 0].min()
            hig = x[0, 0, 0].max()

            axs[0, 0].imshow(r(x[0, 0, 0]).detach().cpu().numpy(), cmap='gray', vmin=low, vmax=hig)
            axs[0, 0].set_title(f'x')
            axs[0, 0].axis('off')

            axs[1, 0].imshow(r(fx[0, 0]).detach().cpu().numpy(), cmap='viridis')
            axs[1, 0].set_title(f'f(x)')
            axs[1, 0].axis('off')
            axs[2, 0].axis('off')

            for i, g in enumerate([g1, g2, g3]):
                axs[0, i + 1].imshow(r(g[0][0, 0, 0]).detach().cpu().numpy(), cmap='gray', vmin=low, vmax=hig)
                axs[0, i + 1].set_title(f'{g[3]} x')
                axs[0, i + 1].axis('off')

                axs[1, i + 1].imshow(r(g[1][0, 0]).detach().cpu().numpy(), cmap='viridis')
                axs[1, i + 1].set_title(f'f({g[3]} x)')
                axs[1, i + 1].axis('off')

                axs[2, i + 1].imshow(r(g[2][0, 0]).detach().cpu().numpy(), cmap='viridis')
                axs[2, i + 1].set_title(f'{g[3]} f(x)')
                axs[2, i + 1].axis('off')

            # Plot original chart
            """
            axs[1, 1].imshow(gx[b, n, d].detach().cpu().numpy(), cmap='viridis')
            axs[1, 1].set_title(f'g * x')
            axs[1, 1].axis('off')

            # Plot original chart
            axs[0, d].imshow(org_org[b, n, d].detach().cpu().numpy(), cmap='viridis')
            axs[0, d].set_title(f'f(x)')
            axs[0, d].axis('off')

            # Plot original chart
            axs[1, d].imshow(original_charts[b, n, d].detach().cpu().numpy(), cmap='viridis')
            axs[1, d].set_title(f'g * f(x)')
            axs[1, d].axis('off')

            # Plot transformed chart
            axs[2, d].imshow(transformed_charts[b, n, d].detach().cpu().numpy(), cmap='viridis')
            axs[2, d].set_title(f'f(g * x)')
            axs[2, d].axis('off')
            """

            plt.tight_layout()
            plt.savefig("test.svg")
            plt.show()

        return pred.loss(y_atlas_true, g_y_atlas)

    # called by LocalTrainer during training
    def regularization(self, _epoch_num):
        # aim for as 'orthogonal' as possible basis matrices
        sim = self.similarity_loss(self.lie_basis)

        # past a certain point, increasing the basis means nothing
        # we only want to increase to a certain extent
        clipped = self.lie_basis.clamp(-self.r2, self.r2)
        trace = torch.sqrt(torch.einsum('kdf,kdf->k', clipped, clipped))
        lie_mag = -torch.mean(trace)

        return self.r1 * sim + self.r3 * lie_mag
