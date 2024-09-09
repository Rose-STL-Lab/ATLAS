import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset, get_ico_timestamp_dataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_iou_perClass
import einops
from icoCNN import *
from ff import R2FeatureField
from os import path
from torch.utils.data import DataLoader
from torch.optim import Adam
import math
import tqdm
import numpy as np

device = get_device()

IN_RAD = 200
OUT_RAD = 150
ICO_RES = 6

# rather naive atlas (not even an atlas in this case): just three charts along equator
class ClimateFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data)

        c = self.data.shape[-1]
        r = self.data.shape[-2]
        mid_c = self.data.shape[-1] // 2
        locs = [(r * 0.4, c * 0.5), (r * 0.5, c * 0.5), (r * 0.6, c * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]

class ClimatePredictor(torch.nn.Module, Predictor):
    def __init__(self, config):
        super().__init__()

        # predictor for each chart
        self.network1 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.network2 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.network3 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.optimizer = Adam(self.parameters(), lr=1e-3)   

    def run(self, x):
        chart_ret = []
        for i, net in enumerate([self.network1, self.network2, self.network3]):
            ret = net(x[:, i])

            # clip to the out radius in the case that it's smaller than in radius
            mid_r = x.shape[-2] // 2
            mid_c = x.shape[-1] // 2
            ret = ret[:, :, mid_r - OUT_RAD : mid_r + OUT_RAD + 1, mid_c - OUT_RAD : mid_c + OUT_RAD + 1]

            chart_ret.append(ret)

        return torch.stack(chart_ret, dim=1)
    
    def loss(self, y_pred, y_true):
        y_pred = y_pred.permute(0, 1, 3, 4, 2).flatten(0, 3)
        y_true = y_true.permute(0, 1, 3, 4, 2).flatten(0, 3)
        return nn.functional.cross_entropy(y_pred, y_true)
        # return jaccard_loss(y_pred, y_true)

    def name(self):
        return "climate" 

    def needs_training(self):
        return True

    def returns_logits(self):
        return True

class ClimateTorchDataset:
    def __init__(self, path, config):
        self.dataset = ClimateDatasetLabeled(path, config)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        h, w = y.shape

        y_onehot = torch.zeros((3, *y.shape), device=device)
        i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y_onehot[torch.tensor(y.values), i, j] = 1
        return torch.tensor(x.values).to(device).squeeze(0), y_onehot

class ClimateIcoDataset:
    def __init__(self, path, config):
        self.dataset = ClimateDatasetLabeled(path, config)

        # dataset has linear latitude and longtitude
        num_lat = 768
        num_lon = 1152
        grid = torch.tensor(icosahedral_grid_coordinates(ICO_RES))

        self.lat = (torch.acos(grid[..., 2]) * num_lat / math.pi).round().long().clip(0, num_lat - 1)
        self.lon = ((math.pi + torch.atan2(grid[..., 1], grid[..., 0])) * num_lon / (2 * math.pi)).round().long().clip(0, num_lon - 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        h, w = y.shape

        y_onehot = torch.zeros((3, *y.shape), device=device)
        i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y_onehot[torch.tensor(y.values), i, j] = 1

        x_ico = torch.tensor(x.values, device=device).squeeze(0)[..., self.lat, self.lon]
        y_ico = y_onehot[..., self.lat, self.lon]
        timestamp = x['time'].values[0]

        return x_ico.unsqueeze(-4), y_ico, timestamp


# adapted from https://github.com/DavidDiazGuerra/icoCNN/blob/master/icoCNN/icoCNN.py
class PadIco(torch.nn.Module):
    """  Pytorch module to pad every chart of an icosahedral signal
    icoCNN.ConvIco already incorporates this padding, so you probably don't want to directly use this class.

    Parameters
    ----------
    r : int
        Resolution of the input icosahedral signal
    R : int, 1 or 6
        6 when the input signal includes the 6 kernel orientation channels or 1 if it doesn't
    smooth_vertices : bool (optional)
        If False (default), the vertices of the icosahedral grid are turned into 0 as done in the original paper by
        Cohen et al. If True, the vertices are replaced by the mean of their neighbours (also equivariant).
    preserve_vertices : bool (optional)
        If True, it avoids turning the vertices into 0 (not equivariant). Default is False.

    Shape
    -----
    Input : [..., R, 5, 2^r, 2^(r+1)]
    Output : [..., R, 5, 2^r+2, 2^(r+1)+2]
    """
    def __init__(self, r, R, smooth_vertices=False, preserve_vertices=False):
        super().__init__()
        self.R = R
        self.r = r
        self.H = 2**r
        self.W = 2**(r+1)

        self.smooth_vertices = smooth_vertices
        if not preserve_vertices:
            self.process_vertices = SmoothVertices(r) if smooth_vertices else CleanVertices(r)
        else:
            assert not smooth_vertices
            self.process_vertices = lambda x: x

        idx_in= torch.arange(R * 5 * self.H * self.W, dtype=torch.long).reshape(R, 5, self.H, self.W)
        idx_out = torch.zeros((R, 5, self.H + 2, self.W + 2), dtype=torch.long)
        idx_out[..., 1:-1, 1:-1] = idx_in
        idx_out[..., 0, 1:2 ** r + 1] = idx_in.roll(1, -3)[..., -1, 2 ** r:]
        idx_out[..., 0, 2 ** r + 1:-1] = idx_in.roll(1, -3).roll(-1, -4)[..., :, -1].flip(-1)
        idx_out[..., -1, 2:2 ** r + 2] = idx_in.roll(-1, -3).roll(-1, -4)[..., :, 0].flip(-1)
        idx_out[..., -1, 2 ** r + 1:-1] = idx_in.roll(-1, -3)[..., 0, 0:2 ** r]
        idx_out[..., 1:-1, 0] = idx_in.roll(1, -3).roll(1, -4)[..., -1, 0:2 ** r].flip(-1)
        idx_out[..., 2:, -1] = idx_in.roll(-1, -3).roll(1, -4)[..., 0, 2 ** r:].flip(-1)
        self.reorder_idx = idx_out

    def forward(self, x):
        x = self.process_vertices(x)
        if self.smooth_vertices:
            smooth_north_pole = einops.reduce(x[..., -1, 0], '... R charts -> ... 1 1', 'mean')
            smooth_south_pole = einops.reduce(x[..., 0, -1], '... R charts -> ... 1 1', 'mean')

        x = einops.rearrange(x, '... R charts H W -> ... (R charts H W)', R=self.R, charts=5, H=self.H, W=self.W)
        y = x[..., self.reorder_idx]

        if self.smooth_vertices:
            y[..., -1, 1] = smooth_north_pole
            y[..., 1, -1] = smooth_south_pole

        return y

GLR = 6
class StrideConv(nn.Module):
    def __init__(self, use_gl, r, Cin, Cout, Rin, bias=True, smooth_vertices=False, stride=1):
        super().__init__()
        assert Rin in [1, 6, 12]
        self.use_gl = use_gl
        self.r = r
        self.Cin = Cin
        self.Cout = Cout
        self.Rin = Rin
        self.stride = stride
        self.Rout = GLR if use_gl else 6
        # scale factor for generating kernels
        self.scale = 0.5

        rp = r if self.stride == 1 else r - 1
        self.process_vertices = SmoothVertices(rp) if smooth_vertices else CleanVertices(rp)
        self.padding = PadIco(r, Rin, smooth_vertices=smooth_vertices)

        s = math.sqrt(2 / (3 * 3 * Cin * Rin))
        self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, 7)))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(Cout))
        else:
            self.register_parameter('bias', None)

        self.kernel_expansion_idx = torch.zeros((Cout, self.Rout, Cin, Rin, 9, 4), dtype=int)
        self.kernel_expansion_idx[..., 0] = torch.arange(Cout).reshape((Cout, 1, 1, 1, 1))
        self.kernel_expansion_idx[..., 1] = torch.arange(Cin).reshape((1, 1, Cin, 1, 1))

        self.kernel_expansion_idx2 = self.kernel_expansion_idx.clone()

        idx_r = torch.arange(0, Rin)
        if use_gl:
            idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                                  (5, 4, -1, 6, 0, 3, -1, 1, 2),
                                  (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                  (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                  (1, 6, -1, 2, 0, 5, -1, 3, 4),
                                  (1, 6, -1, 2, 0, 5, -1, 3, 4)))

            idx_k2 = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                                   (0, 0, -1, 0, 0, 0, -1, 0, 0),
                                   (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                   (0, 0, -1, 0, 0, 0, -1, 0, 0),
                                   (1, 6, -1, 2, 0, 5, -1, 3, 4),
                                   (0, 0, -1, 0, 0, 0, -1, 0, 0)))
        else:
            idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                                  (4, 3, -1, 5, 0, 2, -1, 6, 1),
                                  (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                  (2, 1, -1, 3, 0, 6, -1, 4, 5),
                                  (1, 6, -1, 2, 0, 5, -1, 3, 4),
                                  (6, 5, -1, 1, 0, 4, -1, 2, 3)))
            idx_k2 = idx_k

        for i in range(self.Rout):
            self.kernel_expansion_idx[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
            self.kernel_expansion_idx[:, i, :, :, :, 3] = idx_k[i,:]

            self.kernel_expansion_idx2[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
            self.kernel_expansion_idx2[:, i, :, :, :, 3] = idx_k2[i,:]
            idx_r = idx_r.roll(1)

    def extra_repr(self):
        return "r={}, Cin={}, Cout={}, Rin={}, Rout={}, bias={}"\
            .format(self.r, self.Cin, self.Cout, self.Rin, self.Rout, self.bias is not None)

    def get_kernel(self):
        kernel = self.weight[self.kernel_expansion_idx[..., 0],
                             self.kernel_expansion_idx[..., 1],
                             self.kernel_expansion_idx[..., 2],
                             self.kernel_expansion_idx[..., 3]]
        kernel = kernel.reshape((self.Cout, self.Rout, self.Cin, self.Rin, 3, 3))
        kernel[..., 0, 2] = 0
        kernel[..., 2, 0] = 0

        kernel2 = self.weight[self.kernel_expansion_idx2[..., 0],
                              self.kernel_expansion_idx2[..., 1],
                              self.kernel_expansion_idx2[..., 2],
                              self.kernel_expansion_idx2[..., 3]]
        kernel2 = kernel2.reshape((self.Cout, self.Rout, self.Cin, self.Rin, 3, 3))
        kernel2[..., 0, 2] = 0
        kernel2[..., 2, 0] = 0

        return kernel * self.scale + kernel2 * (1 - self.scale)

    def forward(self, x):
        x = self.padding(x)
        x = einops.rearrange(x, '... C R charts H W -> ... (C R) (charts H) W', C=self.Cin, R=self.Rin, charts=5)
        if x.ndim == 3:
            x = x.unsqueeze(0)
            remove_batch_size = True
        else:
            remove_batch_size = False
            batch_shape = x.shape[:-3]
            x = x.reshape((-1,) + x.shape[-3:])

        kernel = self.get_kernel()
        kernel = einops.rearrange(kernel, 'Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk', Hk=3, Wk=3)
        bias = einops.repeat(self.bias, 'Cout -> (Cout Rout)', Cout=self.Cout, Rout=self.Rout) \
            if self.bias is not None else None

        y = torch.nn.functional.conv2d(x, kernel, bias, padding=(1, 1), stride=self.stride)
        y = einops.rearrange(y, '... (C R) (charts H) W -> ... C R charts H W', C=self.Cout, R=self.Rout, charts=5)
        y = y[..., 1:-1, 1:-1]
        if remove_batch_size: y = y[0, ...]
        else: y = y.reshape(batch_shape + y.shape[1:])

        if self.stride == 2:
            flat_y = y.flatten(0, 2)
            flat_y = torch.nn.functional.pad(flat_y, (0,1,0,1), mode='replicate')
            y = flat_y.unflatten(0, y.shape[:3])
            return self.process_vertices(y)
        else:
            return self.process_vertices(y)

            
class GaugeDownLayer(nn.Module):
    def __init__(self, use_gl, r, c_in, c_out, r_in):
        super().__init__()

        self.model = nn.Sequential(
            StrideConv(use_gl, r, c_in, c_out, r_in, stride=2),
            LNormIco(c_out, GLR if use_gl else 6),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class GaugeUpLayer(nn.Module):
    def __init__(self, use_gl, r, old_c_in, c_in, c_out, activate=True):
        super().__init__()

        c = GLR if use_gl else 6
        self.model = nn.Sequential(
            StrideConv(use_gl, r + 1, old_c_in + c_in, c_out, c),
            LNormIco(c_out, c),
        )
        self.lnorm = LNormIco(c_out, c)
        self.activate = activate

    def forward(self, old, x):
        flat_x = x.flatten(0, 2)
        upsampled = torch.nn.functional.interpolate(flat_x, scale_factor=2, mode='bilinear')
        upsampled = upsampled.unflatten(0, x.shape[:3])

        if old is not None:
            full_input = torch.cat((old, upsampled), dim=1)
        else:
            full_input = upsampled
        ret = self.model(full_input)
        if self.activate:
            ret = torch.nn.functional.relu(ret)
        return ret


class GaugeEquivariantCNN(nn.Module):
    def __init__(self, use_gl):
        super().__init__()

        r = ICO_RES

        def c(raw):
            return raw
            return int(raw / math.sqrt(2))

        self.d1 = GaugeDownLayer(use_gl, r - 0,  4, c(16), 1)
        self.d2 = GaugeDownLayer(use_gl, r - 1, c(16), c(32), GLR)
        self.d3 = GaugeDownLayer(use_gl, r - 2, c(32), c(64), GLR)
        self.d4 = GaugeDownLayer(use_gl, r - 3, c(64), c(128), GLR)

        self.u4 = GaugeUpLayer(use_gl, r - 4, c(64), c(128), c(64))
        self.u3 = GaugeUpLayer(use_gl, r - 3, c(32), c(64), c(32))
        self.u2 = GaugeUpLayer(use_gl, r - 2, c(16), c(32), c(16))
        self.u1 = GaugeUpLayer(use_gl, r - 1, 0, c(16), 3, activate=False)


    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u4 = self.u4(d3, d4)
        u3 = self.u3(d2, u4)
        u2 = self.u2(d1, u3)
        u1 = self.u1(None, u2)

        # collapse 6 orientations
        return torch.sum(u1, dim=-4)

def discover():
    train_path = './data/climate'

    config = Config(bs=4)
    config.fields = {"TMQ": {"mean": 19.21859, "std": 15.81723}, 
                     "U850": {"mean": 1.55302, "std": 8.29764},
                     "V850": {"mean": 0.25413, "std": 6.23163},
                     "PSL": {"mean": 100814.414, "std": 1461.2227} 
                    }
    config.label_length = 3 # nothing, AR, TC
    config.field_length = len(config.fields)

    if config.reuse_predictor:
        predictor = torch.load('predictors/climate.pt')
    else:
        predictor = ClimatePredictor(config)
    
    basis = GroupBasis(
        config.field_length, 2, config.label_length, 4, config.standard_basis, 
        lr=5e-4, in_rad=IN_RAD, out_rad=OUT_RAD, 
        identity_in_rep=True,
        identity_out_rep=True, out_interpolation='nearest', r3=5.0
    )

    dataset = ClimateTorchDataset(path.join(train_path, 'train'), config)

    gdn = LocalTrainer(ClimateFeatureField, predictor, basis, dataset, config)   
    gdn.train()

def dataset_iou(dataset):
    bg_iou = []
    tc_iou = []
    ar_iou = []
    for _, v in dataset.items():
        if len(v) == 1:
            continue

        cm = torch.zeros((3, 3), device=device)
        count = 0
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                _, x = torch.max(v[i], dim=0)
                _, y = torch.max(v[j], dim=0)

                for r in range(3):
                    for c in range(3):
                        cm[r, c] += torch.sum((x == r) & (y == c))
                count += x.numel()

        i, j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        bg = float(cm[0, 0] / torch.sum(cm[(i == 0) | (j == 0)]).detach().cpu())
        # avoid nans
        if bg == bg:
            bg_iou.append(bg)
        tc = float(cm[1, 1] / torch.sum(cm[(i == 1) | (j == 1)]).detach().cpu())
        if tc == tc:
            tc_iou.append(tc)
        ar = float(cm[2, 2] / torch.sum(cm[(i == 2) | (j == 2)]).detach().cpu())
        if ar == ar:
            ar_iou.append(ar)

    bg_iou = np.mean(bg_iou)
    tc_iou = np.mean(tc_iou)
    ar_iou = np.mean(ar_iou)

    iou = np.mean([bg_iou, tc_iou, ar_iou])
    print("dataset ious: bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou)


def train(use_gl):
    print("Using gl model:", use_gl)
    train_path = './data/climate/train'
    test_path = './data/climate/test'

    config = Config()
    config.fields = {"TMQ": {"mean": 19.21859, "std": 15.81723},
                     "U850": {"mean": 1.55302, "std": 8.29764},
                     "V850": {"mean": 0.25413, "std": 6.23163},
                     "PSL": {"mean": 100814.414, "std": 1461.2227}
                    }
    # from https://github.com/andregraubner/ClimateNet/blob/main/config.json
    config.labels = ["Background", "Tropical Cyclone", "Atmospheric River"]
    config.label_length = 3
    config.field_length = len(config.fields)
    config.lr = 0.0005

    train_dataset = ClimateIcoDataset(train_path, config)
    test_dataset = ClimateIcoDataset(test_path, config)

    date_train_dataset = get_ico_timestamp_dataset(train_dataset)
    date_test_dataset = get_ico_timestamp_dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    def print_iou(cm):
        i, j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        bg_iou = float(cm[0, 0] / torch.sum(cm[(i == 0) | (j == 0)]).detach().cpu())
        tc_iou = float(cm[1, 1] / torch.sum(cm[(i == 1) | (j == 1)]).detach().cpu())
        ar_iou = float(cm[2, 2] / torch.sum(cm[(i == 2) | (j == 2)]).detach().cpu())
        denom = torch.max(cm.sum(dim=1), torch.tensor(1))
        precision = float((cm[[0, 1, 2], [0, 1, 2]] / denom).mean().detach().cpu())
        iou = torch.tensor([bg_iou, tc_iou, ar_iou]).mean()
        print("bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou, "precision", precision)
        print("confusion matrix:\n", cm)


    model = GaugeEquivariantCNN(use_gl).to(device)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    for e in range(config.epochs):
        losses = []

        cm = torch.zeros((3, 3), device=device)
        for xx, y_true, timestamps in tqdm.tqdm(train_loader):
            y_pred = model(xx)

            _, y_pred_ind = torch.max(y_pred, dim=1)
            _, y_true_ind = torch.max(y_true, dim=1)

            for r in range(3):
                for c in range(3):
                    cm[r, c] += torch.sum((y_true_ind == r) & (y_pred_ind == c))

            loss = jaccard_loss(y_pred.flatten(2, 3).cpu(), y_true_ind.flatten(1, 2).cpu())
            # loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            losses.append(float(loss.detach().cpu()))

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        print("Epoch", e, "Loss", np.mean(losses), "IOUs")
        print_iou(cm)

    model.eval()

    bg_iou = []
    tc_iou = []
    ar_iou = []
    precision = []
    for x, ys in tqdm.tqdm(date_test_dataset.values()):
        y_pred = model(x.unsqueeze(0))
        _, y_pred_ind = torch.max(y_pred, dim=1)

        cm = torch.zeros((3, 3), device=device)
        for y in ys:
            _, y_true_ind = torch.max(y.unsqueeze(0), dim=1)

            for r in range(3):
                for c in range(3):
                    cm[r, c] += torch.sum((y_true_ind == r) & (y_pred_ind == c))

        i, j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        bg = float(cm[0, 0] / torch.sum(cm[(i == 0) | (j == 0)]).detach().cpu())
        # avoid nans
        if bg == bg:
            bg_iou.append(bg)
        tc = float(cm[1, 1] / torch.sum(cm[(i == 1) | (j == 1)]).detach().cpu())
        if tc == tc:
            tc_iou.append(tc)
        ar = float(cm[2, 2] / torch.sum(cm[(i == 2) | (j == 2)]).detach().cpu())
        if ar == ar:
            ar_iou.append(ar)

        
        denom = torch.max(cm.sum(dim=1), torch.tensor(1))
        precision.append(float((cm[[0, 1, 2], [0, 1, 2]] / denom).mean().detach().cpu()))

    bg_iou = np.mean(bg_iou)
    tc_iou = np.mean(tc_iou)
    ar_iou = np.mean(ar_iou)
    precision = np.mean(precision)

    iou = np.mean([bg_iou, tc_iou, ar_iou])
    print("test ious: bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou, "precision", precision)


if __name__ == '__main__':
    #discover()

    # use_gl = True
    train(True)

