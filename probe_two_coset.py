import torch
import tqdm
import numpy as np
from utils import get_device, in_lie_algebra, rmse
from atlasd import Predictor, GlobalTrainer
from group_basis import GlobalGroupBasis
from config import Config

device=get_device()

def fn(u):
    x = u[..., 0, 0]
    y = u[..., 0, 1]
    return torch.atan((y + 0.1) / x).unsqueeze(-1)

class TCPredictor(Predictor):
    def run(self, x):
        return fn(x)

    def name(self):
        return "TC"

    def loss(self, y_pred, y_true):
        return torch.abs(y_pred, y_true).mean()

    def batched_loss(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true), dim=[-1, -2])

    def needs_training(self):
        return False

class TwoCoset(torch.utils.data.Dataset):
    def __init__(self, N=10000):
        super().__init__()
        self.len = N
        self.X = torch.empty((N, 1, 2))
        torch.nn.init.normal_(self.X)
        self.Y = fn(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

if __name__ == '__main__':
    c = Config()
    dataset = TwoCoset(N=c.N)

    pred = TCPredictor()
    basis = GlobalGroupBasis(2, 1, False, num_cosets=64)
    trainer = GlobalTrainer(pred, basis, dataset, c)
    
    lie = torch.tensor([
            [[1, 0], [0, 1.]]
    ], device=device)
    trainer.discover_cosets(lie, 24)

