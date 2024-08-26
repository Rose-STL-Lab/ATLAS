# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tqdm
from genetic import Genetic
from utils import get_device 
from local_symmetry import Predictor 
from config import Config

device = get_device()
    
class ClassPredictor(Predictor):
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
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def name(self):
        return "toptag"

    def run(self, x):
        ret = self.model(x.flatten(-2))
        return ret

    def loss(self, xx, yy):
        return nn.functional.cross_entropy(xx, yy)

    def needs_training(self):
        return True

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=2, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]
        
        self.y = torch.LongTensor(df[:, -1]).to(device)

    def __len__(self):
        return min(100, self.len)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    n_dim = 4
    n_component = 30
    n_class = 2

    config = Config()

    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if config.reuse_predictor:
        predictor = torch.load('predictors/toptag.pt')
    else:
        exit(1)
        predictor = ClassPredictor(n_dim, n_component, n_class)

        for e in range(config.epochs):
            p_losses = []
            for xx, yy in tqdm.tqdm(loader):
                y_pred = predictor.run(xx)
                y_true = yy

                p_loss = predictor.loss(y_pred, y_true)
                p_losses.append(float(p_loss.detach().cpu()))

                predictor.optimizer.zero_grad()
                p_loss.backward()
                predictor.optimizer.step()

            p_losses = np.mean(p_losses) if len(p_losses) else 0
            torch.save(predictor, "predictors/" + predictor.name() + '.pt')
            print("Epoch", e, "Predictor loss", p_losses) 

    # discover infinitesimal generators via gradient descent

    # discover discrete generators via genetic algorithm
    def score(matrices):
        ret = torch.zeros(matrices.shape[:2], device=device)
        # matrices[:] = torch.eye(4)

        for xx, yy in tqdm.tqdm(loader):
            g_x = torch.einsum('psij, bcj -> psbci', matrices, xx)
            y_pred = predictor.run(g_x)

            identity = matrices.clone()
            identity[:] = torch.eye(4)
            x = torch.einsum('psij, bcj -> psbci', identity, xx)
            y_true = predictor.run(x)

            y_pred = y_pred.permute(2, 3, 0, 1)
            _, y_pind = torch.max(y_pred, dim=1)

            y_true = y_true.permute(2, 3, 0, 1)
            _, y_tind = torch.max(y_true, dim=1)

            ret += (y_pind == y_tind).sum().float() / y_pind.numel() / len(loader)

            # y_true = y_true.permute(2, 0, 1).expand(y_pred.shape[0], y_pred.shape[2], y_pred.shape[3])
            # ret += torch.nn.functional.cross_entropy(y_pred, y_true) / len(loader)
            
        # minimize not maximize
        return -ret

    genetic = Genetic(score, config.num_pops, config.pop_size, 4)
    genetic.run(config.epochs)
