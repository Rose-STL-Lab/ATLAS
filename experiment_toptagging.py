# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)
import sys
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import time
import pandas as pd
import numpy as np
import tqdm
from utils import get_device
from local_symmetry import Predictor
from config import Config

device = get_device(no_mps=True)

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

    def loss(self, y_pred, y_true):
        return nn.functional.cross_entropy(y_pred.squeeze(-1), y_true.squeeze(-1))

    def needs_training(self):
        return True


class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=2, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4 * n_component]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]

        self.y = torch.LongTensor(df[:, -1]).to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == '__main__':
    torch.set_printoptions(precision=9, sci_mode=False)

    n_dim = 4
    n_component=30
    n_class = 2
    max_discrete = 4
    ortho_factor = 0.1
    growth_factor = 1

    config = Config()

    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if config.reuse_predictor:
        predictor = torch.load('predictors/toptag.pt')
    else:
        print("Training Predictor (once finished, run with --reuse_predictor to discover with predictor)")
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

        print("Finished training predictor. Rerun using --reuse_predictor to discover symmetry")
        sys.exit(0)

    # discover infinitesimal generators via gradient descent
    if not config.skip_continuous:
        lie = torch.nn.Parameter(torch.empty(7, 4, 4))
        torch.nn.init.normal_(lie, 0, 0.02)
        optimizer = torch.optim.Adam([lie])

        for e in range(config.epochs):
            average_loss = []
            for xx, yy in tqdm.tqdm(loader):
                coeff = torch.normal(0, 1, (xx.shape[0], lie.shape[0])).to(device) 
                sampled_lie = torch.sum(lie * coeff.unsqueeze(-1).unsqueeze(-1), dim=-3)

                g = torch.matrix_exp(sampled_lie)
                g_x = torch.einsum('bij, bcj -> bci', g, xx)

                # b, 2
                y_pred = predictor.run(g_x)
                # with respect to yy or predictor.run(xx) aren't that different
                y_tind = yy

                loss = torch.nn.functional.cross_entropy(y_pred, y_tind)

                # reg
                trace = torch.einsum('kdf,kdf->k', lie, lie)
                mag = torch.sqrt(trace / lie.shape[1])
                norm = lie / mag.unsqueeze(-1).unsqueeze(-1)

                if config.standard_basis:
                    norm = torch.abs(norm)
                reg = growth_factor * -torch.mean(mag) + ortho_factor * torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', norm, norm), diagonal=1)))

                loss = loss + reg

                average_loss.append(loss.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            average_loss = np.mean(average_loss)
            print("Loss", average_loss, "Basis", lie.detach().cpu())


    # discover discrete generators 
    matrices = torch.nn.Parameter(torch.zeros(512, 4, 4))
    torch.nn.init.normal_(matrices, 0, 1)
    optimizer = torch.optim.Adam([matrices])

    for e in range(config.epochs):
        average_losses = []
        for xx, yy in tqdm.tqdm(loader):
            det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
            normalized = matrices / (det ** 0.25)
            g_x = torch.einsum('pij, bcj -> pbci', normalized.to(device), xx)
            x = xx.unsqueeze(0).expand(g_x.shape)

            # p b 2
            y_pred = predictor.run(g_x)
            # p b
            y_true = predictor.run(x)
            y_tind = torch.nn.functional.softmax(y_true, dim=-1)

            y_pred = torch.permute(y_pred, (1, 2, 0))
            y_tind = torch.permute(y_tind, (1, 2, 0))

            losses = torch.mean(torch.nn.functional.cross_entropy(y_pred, y_tind, reduction='none'), dim=0)
            loss = torch.mean(losses)
            average_losses.append(losses.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        average_losses = np.mean(average_losses, axis=0)

        min_loss = np.min(average_losses)
        min_index = np.argmin(average_losses)
        det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
        normalized = matrices / (det ** 0.25)
        print("Loss", min_loss, "Best", normalized[min_index].detach())

        if e == config.epochs - 1:
            inds = np.argsort(average_losses)
            normalized = normalized[inds].detach().cpu()

            def relates(a, b):
                diff = torch.linalg.inv(a) @ b
                if torch.det(diff) < 0:
                    return False

                if diff[0][0] < 0:
                    return False

                return True
                """
                # otherwise, check if it preserves the the quadratic on the unit vectors
                # (heuristic, realistically if it passes the first two, they probably relate
                # as we are iterating the matrices in order of score, but solutions are always only approximate)
                def good_column(t, x, y, z):
                    return np.abs((t * t - x * x - y * y - z * z) - 1) < 1e-1

                return np.all([good_column(*column) for column in diff])
                """

            # print out 4 highest matrices that do not relate (by SO+(1, 3)) to one another
            # This dataset does not have any time reversal, so we cannot find P or PT
            # but we are generally able to find P
            final = []

            for mat in normalized:
                if len(final) == max_discrete:
                    break

                for f in final:
                    if relates(f, mat):
                        break
                else:
                    final.append(mat)
            print("2 Best Final Discrete Matrices")
            for tensor in final:
                print(tensor, "determinant", torch.det(tensor))

