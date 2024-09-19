import torch
import torch.nn as nn

from utils import get_device, transform_atlas
device = get_device()


def normalize(x):
    # from lie gan
    trace = torch.einsum('kdf,kdf->k', x, x)
    factor = torch.sqrt(trace / x.shape[1])
    x = x / factor.unsqueeze(-1).unsqueeze(-1)
    return x


class GroupBasis(nn.Module):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, num_cosets=32, 
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
        self.num_cosets = num_cosets
        self.dtype = dtype
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.standard_basis = standard_basis

        self.lie_basis = nn.Parameter(torch.empty((num_basis, man_dim, man_dim), dtype=dtype).to(device))
        self.in_basis = nn.Parameter(torch.empty((num_basis, in_dim, in_dim), dtype=dtype).to(device))
        self.out_basis = nn.Parameter(torch.empty((num_basis, out_dim, out_dim), dtype=dtype).to(device))

        for tensor in [self.in_basis, self.lie_basis, self.out_basis]:
            nn.init.normal_(tensor, 0, 0.02)

        self.cosets = nn.Parameter(torch.empty((num_cosets, man_dim, man_dim), dtype=dtype).to(device))
        nn.init.normal_(self.cosets, 0, 0.02)

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

        coeffs = self.sample_coefficients((bs, x.num_charts())) 

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

        return pred.loss(y_atlas_true, g_y_atlas)

    def coset_step(self, x, pred):
        # for now, can only handle identity in and out rep
        assert self.identity_in_rep and self.identity_out_rep

        bs = x.batch_size()

        coeffs = self.sample_coefficients((bs, x.num_charts())) 

        cosets = self.cosets.tile(bs, 1, 1)
        in_rep = torch.eye(self.in_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs * self.num_cosets, x.num_charts(), 1, 1)
        out_rep = torch.eye(self.out_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs * self.num_cosets, x.num_charts(), 1, 1)

        x_atlas = x.regions(self.in_rad)
        g_x_atlas = transform_atlas(cosets, in_rep, x_atlas, self.in_interpolation)

        y_atlas = pred.run(x_atlas)
        if pred.returns_logits():
            y_atlas = torch.nn.functional.softmax(y_atlas, dim=-3)
        y_atlas = y_atlas.detach()
        g_y_atlas = transform_atlas(cosets, out_rep, y_atlas, self.out_interpolation)

        y_atlas_true = pred.run(g_x_atlas)

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
