import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer  
from bases import VertexBasis
from manifold_ops import *

device = get_device()
class WindingPredictor(Predictor):
    def run(self, x):
        return torch.linalg.vector_norm(x.data, dim=-1, keepdim=True)

    def needs_training(self):
        return False

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, N): 
        self.N = N
        self.tensor = torch.normal(0, 1, (N, 2)).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], WindingPredictor()((self.tensor[index]))

if __name__ == '__main__':
    epochs = 25
    N = 100000
    bs = 64

    predictor = WindingPredictor()
    basis = GroupBasis(1)

    dataset = ToyDataset(N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)

    gdn = LocalTrainer(predictor, basis, only_invariance=False)   
    gdn.train(loader, epochs)

