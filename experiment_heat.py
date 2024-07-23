import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, Homomorphism
from ff_transformer import R1FFTransformer
from config import Config

LINE_LEN = 100
LINE_KEY = 11

device = get_device()

def step(x, dt, alpha=2, beta=1):
    # x: [bs, line_len, 2] 
    # modified heat equation
    # du/dt = alpha * d^2u/dx^2 + beta * d^2v/dx^2
    # dv/dt = alpha * d^2v/dx^2 + beta * d^2u/dx^2
    dudx = x[..., 1:, 0] - x[..., :-1, 0]
    d2udx = dudx[..., 1:] - dudx[..., :-1]
    d2udx = torch.cat((d2udx, d2udx[..., -2:]), dim=-1)

    dvdx = x[..., 1:, 1] - x[..., :-1, 1]
    d2vdx = dvdx[..., 1:] - dvdx[..., :-1]
    d2vdx = torch.cat((d2vdx, d2vdx[..., -2:]), dim=-1)
   
    dudt = alpha * d2udx + beta * d2vdx
    dvdt = alpha * d2vdx + beta * d2udx

    return torch.stack((x[..., 0] + dudt * dt, x[..., 1] + dvdt * dt), dim=-1)

class HeatPredictor(nn.Module, Predictor):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * LINE_LEN, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * LINE_LEN),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def run(self, x):
        return self.model(torch.flatten(x, start_dim=-2)).reshape(x.shape)

    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=-2)).reshape(x.shape)

    def needs_training(self):
        return True

class HeatHomomorphism(nn.Module, Homomorphism):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * LINE_LEN, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * LINE_LEN),
        )

    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=-3)).reshape(x.shape)

class HeatDataset(torch.utils.data.Dataset):
    def __init__(self, N): 
        self.N = N
        lerp = R1FFTransformer(LINE_LEN, 6)

        key = torch.normal(0, 2, (N, 6, 2))
        self.tensor = lerp.smooth_function(key)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], step(self.tensor[index], 0.2)


if __name__ == '__main__':
    config = Config()

    predictor = HeatPredictor()
    transformer = R1FFTransformer(LINE_LEN, LINE_KEY)
    homomorphism = HeatHomomorphism()
    basis = GroupBasis(2, transformer, homomorphism, 1, config.standard_basis, lr=5e-4)
    dataset = HeatDataset(config.N)

    gdn = LocalTrainer(predictor, basis, dataset, config)   
    gdn.train()

