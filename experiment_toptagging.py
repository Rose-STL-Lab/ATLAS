# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)
import sys
import torch
from torch import nn
import pandas as pd
import numpy as np
import tqdm
from utils import get_device
from local_symmetry import Predictor
from config import Config

device = get_device()


class ClassPredictor(nn.Module, Predictor):
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

    def forward(self, x):
        return self.run(x)

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

        print("Min", torch.min(self.X[:, :, 0]))

        self.y = torch.LongTensor(df[:, -1]).to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def lie_algebra(config, predictor, loader):
    ortho_factor = 0.1
    growth_factor = 1

    lie = torch.nn.Parameter(torch.empty(7, 4, 4, device=device))
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
        print("Epoch", e, "Loss", average_loss, "Basis", lie.detach().cpu())


def cosets(config, predictor, loader):
    max_discrete = 16

    # discover discrete generators
    matrices = torch.nn.Parameter(torch.zeros(256, 4, 4, device=device))
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
                # previously discovered result
                # feel free to change with result
                lie = tensor([
                        [[     0.000213433,     -0.005174065,     -0.003704194,
                        0.005485935],
                        [    -0.000220797,      0.007107087,      9.064596176,
                        -0.016680451],
                        [     0.013829147,     -9.083515167,      0.000409880,
                        0.005038944],
                        [    -0.002429933,     -0.023221379,      0.000788841,
                        0.003229556]],

                        [[    -0.001463409,      0.004085749,     -0.001559350,
                        -0.002202956],
                        [    -0.001653554,     -0.007234810,      0.012270011,
                        2.069200516],
                        [     0.001355464,      0.007698952,      0.010395083,
                        -0.015434604],
                        [     0.005240460,     -2.229178905,     -0.014423830,
                        -0.005905334]],

                        [[    -0.000307889,     -0.008264347,     -0.007837607,
                        0.002696112],
                        [     0.009847505,     -0.002272248,      0.009314264,
                        -0.001711847],
                        [     0.008981410,      0.007254289,      0.010668403,
                        -1.946889877],
                        [     0.002110766,     -0.003389482,      2.023861647,
                        -0.012214450]],

                        [[    -0.505111814,     -0.004659682,     -0.004683510,
                        0.003998289],
                        [     0.005436522,     -0.511033416,      0.000928673,
                        -0.001898132],
                        [     0.004792073,      0.000898482,     -0.486769170,
                        -0.009679070],
                        [    -0.002720858,     -0.002526487,     -0.009919690,
                        -0.502397895]],

                        [[    -0.001996114,      0.435598731,     -0.001201532,
                        -0.000132177],
                        [     0.414271683,      0.002702238,      0.005514442,
                        -0.005524699],
                        [     0.001359800,      0.004905965,     -0.002260074,
                        -0.001266959],
                        [     0.000610460,     -0.005306936,     -0.000590180,
                        0.004542169]],

                        [[    -0.001848966,      0.003491087,      0.448010474,
                        0.001348291],
                        [    -0.003070940,     -0.001993307,     -0.010231275,
                        0.000965058],
                        [     0.420198143,     -0.011571770,     -0.000276351,
                        0.000804428],
                        [    -0.000751693,      0.001232286,     -0.000692502,
                        0.005575983]],

                        [[    -0.001681802,     -0.003640521,     -0.000265728,
                        -0.809178114],
                        [     0.002888469,     -0.004787319,     -0.001104475,
                        -0.007128626],
                        [    -0.003361337,     -0.002041987,      0.004072640,
                        0.009291197],
                        [    -0.818222344,     -0.009369194,      0.006323077,
                        0.004909910]]
                ], device=device)

                return in_lie_algebra(diff, lie)

            # print out highest matrices that do not relate (by SO+(1, 3)) to one another
            # This dataset does not have any time reversal, so we cannot find T or PT
            # but we are generally able to find P
            final = []

            for mat in normalized[:max_discrete]:
                for f in final:
                    if relates(f, mat):
                        break
                else:
                    final.append(mat)

            print("Best Final Discrete Matrices")
            for tensor in final:
                print(tensor, "determinant", torch.det(tensor))


def discover(config, continuous, discrete):
    torch.set_printoptions(precision=9, sci_mode=False)

    targets = []
    if continuous:
        targets.append("algebra")
    if discrete:
        targets.append("cosets")

    print("Task: discovering", targets)

    n_dim = 4
    n_component = 30
    n_class = 2

    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if config.reuse_predictor:
        predictor = torch.load('predictors/toptag.pt').to(device)
        print("* Reusing Predictor")
    else:
        print("* Training Predictor")
        predictor = ClassPredictor(n_dim, n_component, n_class).to(device)

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
    if continuous:
        print("* Discovering lie algebra")
        lie_algebra(config, predictor, loader)

    # discrete
    if discrete:
        print("* Discovering cosets")
        cosets(config, predictor, loader)


if __name__ == '__main__':
    c = Config()

    if c.task == 'discover':
        discover(c, True, True)
    elif c.task == 'discover_algebra':
        discover(c, True, False)
    elif c.task == 'discover_cosets':
        discover(c, False, True)
    else:
        print("Unknown task for Top Tagging. Downstream tasks are done in their own directory")
