# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import get_device, affine_coord
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import SingletonFFTransformer

device = get_device()

class AugClassificationModel(nn.Module):
    def __init__(self, n_dim, n_components, n_classes):
        super().__init__()
        self.n_dim = n_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Linear(n_dim * n_components, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )
    def forward(self, x):
        return self.model(x.reshape(-1, self.n_dim * self.n_components))
    
class ClassPredictor(Predictor):
    # def __init__(self, n_dim, n_components, n_classes):
    #     super().__init__()
    #     self.n_dim = n_dim
    #     self.n_components = n_components
    #     self.n_classes = n_classes
    #     self.model = nn.Sequential(
    #         nn.Linear(n_dim * n_components, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, n_classes),
    #     ).to(device)
    # 
    # def run(self, x):
    #     return self.model(x.reshape(-1, self.n_dim * self.n_components).to(device))

    def __init__(self, n_dim, n_channel):
        self.model = AugClassificationModel(n_dim, 2, 2).to(device)
        self.n_dim = n_dim
        self.n_channel = n_channel
        self.Li = nn.Parameter(torch.randn(n_channel, n_dim, n_dim)).to(device)
        self.sigma = nn.Parameter(torch.eye(n_channel, n_channel)).to(device)
        self.mu = nn.Parameter(torch.zeros(n_channel)).to(device)
        self.dummy_pos = None

    def normalize_factor(self):
        trace = torch.einsum('kdf,kdf->k', self.Li, self.Li)
        factor = torch.sqrt(trace / self.Li.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self):
        return self.Li / (self.normalize_factor() + 1e-6)

    # x: (batch_size, n_components, n_dim); y: (batch_size, n_components_y, n_dim)
    def run(self, x):
        if len(x.shape) == 2:
            x.unsqueeze_(1)
        batch_size = x.shape[0]
        #z = self.sample_coefficient(batch_size, x.device)
        z = torch.randn(batch_size, self.n_channel, device=device) @ self.sigma + self.mu
        Li = self.normalize_L()
        g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, Li))
        x_p = affine_coord(torch.einsum('bjk,btk->btj', g_z, x), self.dummy_pos)
        return self.model(x_p)

    def needs_training(self):
        return False

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/val.h5', flatten=False, n_component=3, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = self.X * np.random.uniform(1-noise, 1+noise, size=self.X.shape)
        self.y = df[:, -1]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.y = torch.LongTensor(self.y).to(device)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    epochs = 25
    N = 1000
    bs = 64

    n_dim = 4
    n_component = 2
    n_class = 2
    # n_channel = 7
    # d_input_size = n_dim * n_component
    # emb_size = 32

    # predictor from LieGAN codebase
    predictor = ClassPredictor(n_dim, n_component)

    transformer = SingletonFFTransformer()
    basis = GroupBasis(4, transformer, 6, 3)
    basis.predictor = predictor

    # dataset from LieGAN codebase
    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)