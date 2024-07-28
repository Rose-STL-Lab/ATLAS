import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, Homomorphism
from ff_transformer import S1FFTransformer
from config import Config

LINE_LEN = 100
LINE_KEY = 10

device = get_device()

def step(x, dt):
    # x: [bs, line_len, 2] 
    # du/dt = -v * (du/dx^2 + dv/dx^2)
    # dv/dt = +u * (dv/dx^2 + dv/dx^2)
    u = x[..., 0]
    v = x[..., 1]
    dudx = torch.roll(u, 1, dims=-1) - u
    dvdx = torch.roll(v, 1, dims=-1) - v

    mag = exp(-(dudx * dudx + dvdx * dvdx))
    dudt = mag * -v
    dvdt = mag * u
   
    return torch.stack((u + dudt * dt, v + dvdt * dt), dim=-1)

class HeatPredictor(nn.Module, Predictor):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * LINE_LEN, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
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
        lerp = S1FFTransformer(LINE_LEN, 5)

        key = torch.normal(0, 2, (N, 5, 2))
        self.tensor = lerp.smooth_function(key)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], step(self.tensor[index], 0.2)


if __name__ == '__main__':
    config = Config()

    predictor = HeatPredictor()
    transformer = S1FFTransformer(LINE_LEN, LINE_KEY)
    homomorphism = HeatHomomorphism()
    basis = GroupBasis(2, transformer, homomorphism, 1, config.standard_basis, lr=5e-4)
    dataset = HeatDataset(config.N)

    gdn = LocalTrainer(predictor, basis, dataset, config)   
    gdn.train()

