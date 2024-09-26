import sys
import torch
from torch import nn
import numpy as np
import tqdm
from ff import R2FeatureField
from utils import get_device, rmse
from group_basis import GroupBasis
from local_symmetry import Predictor, LocalTrainer
from config import Config
from pyperlin import FractalPerlin2D


device = get_device()

IN_RAD = 14
OUT_RAD = 6

# solely predict d/dt
def abs_pde(x_in, dx=0.1):
    ddx = (x_in[..., 2:, 1:-1] - x_in[..., :-2, 1:-1]) / (2 * dx)
    ddy = (x_in[..., 1:-1, 2:] - x_in[..., 1:-1, :-2]) / (2 * dx)

    # K4 group symmetry
    ddt = -ddx.abs() + ddy.abs()
    
    # boundary condition
    return torch.nn.functional.pad(ddt, (1, 1, 1, 1))

def heat_pde(x_in, alpha=1, dx=0.1):
    ddx = (x_in[..., 2:, 1:-1] - x_in[..., :-2, 1:-1]) / (2 * dx)
    ddy = (x_in[..., 1:-1, 2:] - x_in[..., 1:-1, :-2]) / (2 * dx)

    dddx = (ddx[..., 2:, 1:-1] - ddx[..., :-2, 1:-1]) / (2 * dx)
    dddy = (ddy[..., 1:-1, 2:] - ddy[..., 1:-1, :-2]) / (2 * dx)

    ddt = (dddx + dddy) * alpha
    
    # boundary condition
    return torch.nn.functional.pad(ddt, (2, 2, 2, 2), mode='replicate')

class SinglePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 1, kernel_size=3, padding='same'),
        ).to(device)


    def forward(self, x):
        return self.model(x)


class PDEFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data) 

        c = self.data.shape[-1]
        r = self.data.shape[-2]
        
        spots = [0.25, 0.42, 0.58, 0.75]
        locs = []
        for i in spots:
            for j in spots:
                locs.append((r * i, c * j))

        self.locs = [(int(r), int(c)) for r, c in locs]


class PDEPredictor(nn.Module, Predictor):
    def __init__(self):
        super().__init__()
        
        self.predictors = torch.nn.ModuleList([SinglePredictor() for _ in range(16)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)   

    def run(self, x):
        chart_ret = []
        for i, net in enumerate(self.predictors):
            ret = net(x[:, i])
            chart_ret.append(ret)

        return torch.stack(chart_ret, dim=1)

    def forward(self, x):
        return self.run(x)

    def loss(self, y_pred, y_true):
        return (y_pred - y_true).abs().mean()

    def batched_loss(self, y_pred, y_true):
        return (y_pred - y_true).flatten(1).abs().mean(dim=1)

    def name(self):
        return "pde" 

    def needs_training(self):
        return True

    def returns_logits(self):
        return False


class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, N):
        super().__init__()

        shape = (N, 128, 128)
        resolutions = [(2 ** i, 2 ** i) for i in range(1, 4)]
        factors = [1, 0.5, 0.25]
        fp = FractalPerlin2D(shape, resolutions, factors)


        self.X = fp().to(device).unsqueeze(1).detach()
        import matplotlib.pyplot as plt
        plt.imshow(self.X[0].swapaxes(0,2))
        plt.show()
        if dataset == 'abs':
            self.Y = abs_pde(self.X)
        elif dataset == 'heat' :
            self.Y = heat_pde(self.X)
        else:
            raise ValueError("Unknown dataset")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def discover(config, algebra, cosets):
    targets = []
    if algebra:
        targets.append("algebra")
    if cosets:
        targets.append("cosets")

    print("Task: discovering", targets)

    if config.reuse_predictor:
        predictor = torch.load("predictors/pde.pt")
    else:
        predictor = PDEPredictor()

    basis = GroupBasis(
        1, 2, 1, 1, config.standard_basis, 
        in_rad=IN_RAD, out_rad=OUT_RAD, 
        num_cosets=32,
        identity_in_rep=True,
        identity_out_rep=True, 
        # a really small value is needed since the pde values themselves are so small
        # that even incorrect symmetries generate small loss values
        # in general, when there's no symmetry at all the model
        # will go in the direction that's closest to symmetry (even if not true invariance)
        # but this can in many times be undesirable
        r3=0.1,
    )

    dataset = PDEDataset(config.pde, config.N)

    gdn = LocalTrainer(PDEFeatureField, predictor, basis, dataset, config)   
    if algebra:
        gdn.train()
    if cosets:
        def relates(a, b):
            inv = a @ torch.inverse(b)
            return torch.linalg.matrix_norm(inv - torch.eye(2, device=device)) < 0.1

        gdn.discover_cosets(relates, 28)


if __name__ == '__main__':
    c = Config()

    if c.task == 'discover':
        discover(c, True, True)
    elif c.task == 'discover_algebra':
        discover(c, True, False)
    elif c.task == 'discover_cosets':
        discover(c, False, True)
    else:
        print("Unknown task for PDE")
