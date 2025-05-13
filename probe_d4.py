import torch
import tqdm
import numpy as np
from utils import get_device, in_lie_algebra, rmse
from local_symmetry import Predictor, GlobalTrainer
from group_basis import GlobalGroupBasis
from config import Config

device=get_device()

def fn(u):
    x = u[..., 0, 0]
    y = u[..., 0, 1]
    return (torch.abs(x) + torch.abs(y)).unsqueeze(-1)


class D4Predictor(Predictor):
    def run(self, x):
        return fn(x)

    def name(self):
        return "D4"

    def loss(self, y_pred, y_true):
        return torch.abs(y_pred, y_true).mean()

    def batched_loss(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true), dim=[-1, -2])

    def needs_training(self):
        return False

class D4(torch.utils.data.Dataset):
    def __init__(self, N=10000):
        super().__init__()
        self.len = N
        self.X = torch.empty((N, 1, 2), device=device)
        torch.nn.init.normal_(self.X)
        self.Y = fn(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    c = Config()
    dataset = D4(N=c.N)

    pred = D4Predictor()
    basis = GlobalGroupBasis(2, 1, False, num_cosets=256)
    trainer = GlobalTrainer(pred, basis, dataset, c)
    
    lie = torch.zeros((0, 2, 2), device=device)
    trainer.discover_cosets(lie, 128)
