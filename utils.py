import torch
import torch.nn.functional as F
import math


def rmse(xx, yy):
    return torch.sqrt(torch.mean(torch.square(xx - yy)))


def mae(xx, yy):
    return torch.mean(torch.abs(xx - yy))


def in_lie_algebra(matrix, basis, absolute=0.1, steps=10000):
    assert steps > 0

    matrix = matrix.detach().clone()
    weights = torch.nn.Parameter(torch.zeros(len(basis), device=basis.device))
    optim = torch.optim.Adam([weights])

    for _ in range(steps):
        sampled = torch.matrix_exp(torch.einsum('w, wij -> ij', weights, basis))
        loss = ((sampled - matrix) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    return loss - absolute < 0

def get_device(no_mps=True):
    if no_mps:
        return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


device = get_device()


def transform_atlas(matrices, ff_matrices, charts, interpolation):
    bs, num_charts, ff_dim, height, width = charts.shape

    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    grid = torch.stack([x, y], dim=-1).float()
    grid = grid - torch.tensor([width/2, height/2], device=device)

    matrices = matrices.view(bs * num_charts, 2, 2)
    ff_matrices = ff_matrices.view(bs * num_charts, ff_dim, ff_dim)

    grid = grid.unsqueeze(0).repeat(bs * num_charts, 1, 1, 1)

    transformed_grid = torch.einsum('bji,bhwj->bhwi', matrices, grid.to(matrices.device))
    transformed_grid = transformed_grid.to(device) + torch.tensor([width/2, height/2], device=device)

    transformed_grid[:, :, :, 0] = transformed_grid[:, :, :, 0] / (width - 1) * 2 - 1
    transformed_grid[:, :, :, 1] = transformed_grid[:, :, :, 1] / (height - 1) * 2 - 1

    charts_reshaped = charts.reshape(bs * num_charts, ff_dim, height, width)

    transformed_charts = F.grid_sample(charts_reshaped, transformed_grid, align_corners=True, padding_mode='border', mode=interpolation)

    transformed_charts = torch.einsum('bij,bjhw->bihw', ff_matrices.to(device), transformed_charts)
    transformed_charts = transformed_charts.view(bs, num_charts, ff_dim, height, width)

    return transformed_charts


class ManifoldLayer(torch.nn.Module):
    def __init__(self, in_field_len, out_field_len, G):
        super().__init__()

        self.G = G

        self.conv = torch.nn.Conv2d(in_field_len, out_field_len, kernel_size=5, stride=5)
        self.batch_norm = torch.nn.BatchNorm2d(out_field_len)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        ff_type = type(x)

        x = x.atlas()

        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)

        x = ff_type(x)

        return x

    def effective_param_count(self):
        return self.kernel.numel() + self.bias.numel() + sum(p.numel() for p in self.batch_norm.parameters())


# conceptually the same idea, but for manifolds
# where atlas function is just the adjacent elements, we have an optimized version
class ManifoldStandardLayer(torch.nn.Module):
    def __init__(self, in_field_len, out_field_len, G, dilation=1):
        super().__init__()

        self.kernel = torch.nn.Parameter(torch.empty(out_field_len, in_field_len, 5, 5, device=device))
        self.bias = torch.nn.Parameter(torch.empty(out_field_len, device=device))

        # from pytorch
        torch.nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.G = G
        self.dilation = dilation
        self.batch_norm = torch.nn.BatchNorm2d(out_field_len)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        ff_type = type(x)

        x = x.data

        if self.G == 'trivial':
            full_kernel = self.kernel
        elif self.G == 'so2':
            # (really C_4)
            full_kernel = torch.zeros_like(self.kernel)
            for i in range(4):
                full_kernel += torch.rot90(self.kernel, k=i, dims=(-2, -1)) / 4
        else:
            raise ValueError()

        x = torch.nn.functional.conv2d(x, full_kernel, self.bias, stride=1, padding=2*self.dilation, dilation=self.dilation)
        x = self.relu(x)
        x = self.batch_norm(x)

        x = ff_type(x)

        return x

    def effective_param_count(self):
        mul = 1
        if self.G == 'trivial':
            mul = 1
        elif self.G == 'so2':
            # a rotationally symmetric kernel only has 7 parameters in effect
            mul = 7 / 25

        return int(self.kernel.numel() * mul + self.bias.numel() + sum(p.numel() for p in self.batch_norm.parameters()))


class ManifoldPredictor(torch.nn.Module):
    def __init__(self, types, ff_type, G='trivial'):
        super().__init__()

        self.ff_type = ff_type
        layers = []
        for i, o, d in types:
            if self.ff_type.has_standard_atlas():
                layers.append(ManifoldStandardLayer(i, o, G, dilation=d).to(device))
            else:
                layers.append(ManifoldLayer(i, o, G).to(device))
                
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        x = self.ff_type(x)
        for layer in self.layers:
            x = layer(x)

        return x.data

    def effective_param_count(self):
        return sum(l.effective_param_count() for l in self.layers)
