import torch
import torch.nn as nn
import einops

from abc import ABC, abstractmethod

from utils import get_device, transform_atlas
device = get_device()


def normalize(x):
    # from lie gan
    trace = torch.einsum('kdf,kdf->k', x, x)
    factor = torch.sqrt(trace / x.shape[1])
    x = x / factor.unsqueeze(-1).unsqueeze(-1)
    return x


# shared code between local and global training
class GroupBasis(nn.Module, ABC):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, num_cosets=64, 
            lr=5e-4, r1=0.05, r2=1, r3=0.35, dtype=torch.float32, 
            identity_in_rep=True, identity_out_rep=True,
    ):
        super().__init__()
    
        self.in_dim = in_dim
        self.man_dim = man_dim
        self.out_dim = out_dim

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

        cosets = torch.empty((num_cosets, man_dim, man_dim), dtype=dtype).to(device)
        nn.init.normal_(cosets, 0, 1)
        self.cosets = nn.Parameter(cosets)

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

    def norm_cosets(self):
        det = torch.abs(torch.det(self.cosets).unsqueeze(-1).unsqueeze(-1))
        return self.cosets / (det ** (1 / self.man_dim))

    # called by Trainer 
    def regularization(self, _epoch_num):
        # aim for as 'orthogonal' as possible basis matrices
        sim = self.similarity_loss(self.lie_basis)

        # past a certain point, increasing the basis means nothing
        # we only want to increase to a certain extent
        clipped = self.lie_basis.clamp(-self.r2, self.r2)
        trace = torch.sqrt(torch.einsum('kdf,kdf->k', clipped, clipped))
        lie_mag = -torch.mean(trace)

        return self.r1 * sim + self.r3 * lie_mag

    @abstractmethod
    def step(self, x, pred, y):
        ...

    @abstractmethod
    def coset_step(self, x, pred, y):
        ...


class LocalGroupBasis(GroupBasis):
    def __init__(
            self, in_dim, man_dim, out_dim, num_basis, standard_basis, 
            in_rad=10, out_rad=5, in_interpolation='bilinear', out_interpolation='bilinear',
        **kwargs
    ):
        super().__init__(in_dim, man_dim, out_dim, num_basis, standard_basis, **kwargs)
    
        self.in_rad = in_rad
        self.out_rad = out_rad

        self.in_interpolation = in_interpolation
        self.out_interpolation = out_interpolation

    def step(self, x, pred, _y):
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

        r = y_atlas_true.shape[-2] // 2
        c = y_atlas_true.shape[-1] // 2
        y_atlas_true = y_atlas_true[..., r - self.out_rad: r + self.out_rad + 1, c - self.out_rad: c + self.out_rad + 1]
        g_y_atlas = g_y_atlas[..., r - self.out_rad: r + self.out_rad + 1, c - self.out_rad: c + self.out_rad + 1]

        return pred.loss(y_atlas_true, g_y_atlas)

    def coset_step(self, x, pred, _y):
        # for now, can only handle identity in and out rep
        assert self.identity_in_rep and self.identity_out_rep

        bs = x.batch_size()

        # technically each chart is transformed the same way, 
        # but we ensure independence through the separate predictors elsewhere so it's fine
        cosets = einops.repeat(self.norm_cosets(), 'c ... -> (c bs) ...', bs=bs * x.num_charts())
        in_rep = torch.eye(self.in_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs * len(self.cosets), x.num_charts(), 1, 1)
        out_rep = torch.eye(self.out_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(bs * len(self.cosets), x.num_charts(), 1, 1)

        x_atlas = einops.repeat(x.regions(self.in_rad), 'bs ... -> (c bs) ...', c=len(self.cosets))
        g_x_atlas = transform_atlas(cosets, in_rep, x_atlas, self.in_interpolation)

        y_atlas = pred.run(x_atlas)
        if pred.returns_logits():
            y_atlas = torch.nn.functional.softmax(y_atlas, dim=-3)
        y_atlas = y_atlas.detach()
        g_y_atlas = transform_atlas(cosets, out_rep, y_atlas, self.out_interpolation)

        y_atlas_true = pred.run(g_x_atlas)

        r = y_atlas_true.shape[-2] // 2
        c = y_atlas_true.shape[-1] // 2
        y_atlas_true = y_atlas_true[..., r - self.out_rad: r + self.out_rad + 1, c - self.out_rad: c + self.out_rad + 1]
        g_y_atlas = g_y_atlas[..., r - self.out_rad: r + self.out_rad + 1, c - self.out_rad: c + self.out_rad + 1]

        return y_atlas_true.unflatten(0, (-1, bs)), g_y_atlas.unflatten(0, (-1, bs))

class GlobalGroupBasis(GroupBasis):

    def __init__(
            self, in_dim, num_basis, standard_basis, 
        **kwargs
    ):
        super().__init__(1, in_dim, 1, num_basis, standard_basis, **kwargs)
    

    def step(self, x, pred, _y):
        assert self.identity_in_rep and self.identity_out_rep

        bs = x.shape[0]

        coeffs = self.sample_coefficients((bs,)) 
        sampled_lie = torch.sum(lie * coeffs.unsqueeze(-1).unsqueeze(-1), dim=-3)

        g = torch.matrix_exp(sampled_lie)
        g_x = torch.einsum('bij, bcj -> bci', g, x)

        y_pred = pred.run(g_x)
        y_tind = pred.run(x)
        if pred.returns_logits():
            y_tind = torch.nn.functional.softmax(y_tind, dim=-1)

        return pred.loss(y_pred, y_tind)

    def coset_step(self, x, pred, _y):
        assert self.identity_in_rep and self.identity_out_rep

        bs = x.shape[0]

        normalized = self.norm_cosets()
        g_x = torch.einsum('pij, bcj -> pbci', normalized, x)
        x = x.unsqueeze(0).expand(g_x.shape)

        # p b 2
        y_pred = pred.run(g_x)
        # p b
        y_tind = pred.run(x)
        if pred.returns_logits():
            y_tind = torch.nn.functional.softmax(y_tind, dim=-1)

        y_pred = torch.permute(y_pred, (1, 2, 0))
        y_tind = torch.permute(y_tind, (1, 2, 0))

        return y_pred, y_tind
