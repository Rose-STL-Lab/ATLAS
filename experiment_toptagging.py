import sys
import torch
from torch import nn
import pandas as pd
import numpy as np
import tqdm
from utils import get_device, in_lie_algebra
from atlasd import Predictor, GlobalTrainer
from group_basis import GlobalGroupBasis
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

    def batched_loss(self, y_pred, y_true):
        y_pred = torch.permute(y_pred, (1, 2, 0))
        y_tind = torch.permute(y_true, (1, 2, 0))

        return torch.mean(torch.nn.functional.cross_entropy(y_pred, y_tind, reduction='none'), dim=0)

    def returns_logits(self):
        return True

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

def discover(config, continuous, discrete):
    targets = []
    if continuous:
        targets.append("algebra")
    if discrete:
        targets.append("cosets")

    print("Task: discovering", targets)

    n_dim = 4
    n_component = 30
    n_class = 2

    predictor = ClassPredictor(n_dim, n_component, n_class).to(device)
    if config.reuse_predictor:
        print("* Reusing Predictor")
        predictor.load_state_dict(torch.load('predictors/toptag.pt', weights_only=True, map_location=device))
    else:
        print("* Training Predictor")

    dataset = TopTagging(n_component=n_component)

    basis = GlobalGroupBasis(
        4, 7, c.standard_basis, num_cosets=64,
        r1=0.1, r3=1
    )
    trainer = GlobalTrainer(predictor, basis, dataset, c)

    if continuous:
        trainer.train()

    if discrete:
        # previously discovered result
        # feel free to change with result
        lie = torch.tensor([
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


        trainer.discover_cosets(lie, 16)


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
