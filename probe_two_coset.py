import torch
import tqdm
import numpy as np
from utils import get_device, in_lie_algebra, rmse
from local_symmetry import Predictor
from config import Config

device=get_device()

def fn(u):
    x = u[..., 0, 0]
    y = u[..., 0, 1]
    return torch.atan((y + 0.1) / x)

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

def cosets(config, loader):
    max_discrete = 24

    # discover discrete generators
    matrices = torch.zeros(64, 2, 2, device=device)
    torch.nn.init.normal_(matrices, 0, 1)
    matrices = torch.nn.Parameter(matrices)

    optimizer = torch.optim.Adam([matrices])

    for e in range(config.epochs):
        average_losses = []
        for xx, yy in tqdm.tqdm(loader):
            det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
            normalized = matrices / (det ** 0.5)
            g_x = torch.einsum('pij, bcj -> pbci', normalized.to(device), xx)
            x = xx.unsqueeze(0).expand(g_x.shape)

            # p b 2
            y_pred = fn(g_x)
            # p b
            y_true = fn(x)

            losses = torch.mean(torch.abs(y_pred - y_true), dim=-1)
            loss = torch.mean(losses)
            average_losses.append(losses.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_losses = np.mean(average_losses, axis=0)

        min_loss = np.min(average_losses)
        min_index = np.argmin(average_losses)
        det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
        normalized = matrices / (det ** 0.5)
        print("Loss", min_loss, "Best", normalized[min_index].detach())

        if e == config.epochs - 1:
            inds = np.argsort(average_losses)
            normalized = normalized[inds].detach().cpu()

            def relates(a, b):
                diff = torch.linalg.inv(a) @ b
                lie = torch.tensor([
                        [[1, 0], [0, 1.]]
                ], device=diff.device)

                return in_lie_algebra(diff, lie)

            # print out highest matrices that do not relate (by SO+(1, 3)) to one another
            # This dataset does not have any time reversal, so we cannot find T or PT
            # but we are generally able to find P
            final = []

            negs = 0
            for i, mat in enumerate(normalized[:max_discrete]):
                print(mat)
                if mat[0][0] < 0:
                    negs += 1
                for f in final:
                    if relates(f[0], mat):
                        break
                else:
                    final.append((mat, average_losses[inds[i]]))

            print("negs", negs, "pos", max_discrete - negs)
            print("Best Final Discrete Matrices")
            for tensor in final:
                print(tensor)

if __name__ == '__main__':
    c = Config()
    dataset = TwoCoset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    cosets(c, loader)
