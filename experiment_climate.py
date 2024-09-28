import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, get_ico_timestamp_dataset
from climatenet.models import CGNetModule
from climatenet.utils.losses import jaccard_loss
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


# rather naive atlas (not even an atlas in this case): just four charts along equator
class ClimateFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data)

        c = self.data.shape[-1]
        r = self.data.shape[-2]
        locs = [(r * 0.35, c * 0.5), (r * 0.45, c * 0.5), (r * 0.55, c * 0.5), (r * 0.65, c * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]


class ClimatePredictor(torch.nn.Module, Predictor):
    def __init__(self, config):
        super().__init__()

        # predictor for each chart
        self.network1 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.network2 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.network3 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.network4 = CGNetModule(False, classes=config.label_length, channels=config.field_length).to(device)
        self.optimizer = Adam(self.parameters(), lr=1e-3)   

    def run(self, x):
        chart_ret = []
        for i, net in enumerate([self.network1, self.network2, self.network3, self.network4]):
            ret = net(x[:, i])

            # clipping to out radius is done by group basis

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


class StrideConv(nn.Module):
    def __init__(self, kernel_type, r, Cin, Cout, Rin, bias=True, smooth_vertices=False, stride=1):
        super().__init__()
        assert Rin in [1]
        self.r = r
        self.Cin = Cin
        self.Cout = Cout
        self.Rin = Rin
        self.stride = stride
        self.Rout = 1 if kernel_type == 'discovered' else 6

        rp = r if self.stride == 1 else r - 1
        self.process_vertices = SmoothVertices(rp) if smooth_vertices else CleanVertices(rp)
        self.padding = PadIco(r, Rin, smooth_vertices=smooth_vertices)

        s = math.sqrt(2 / (3 * 3 * Cin * Rin))
        n = 1 if kernel_type == 'discovered' else 7
        self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, n)))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(Cout))
        else:
            self.register_parameter('bias', None)

        self.kernel_expansion_idx = torch.zeros((Cout, self.Rout, Cin, Rin, 9, 4), dtype=int)
        self.kernel_expansion_idx[..., 0] = torch.arange(Cout).reshape((Cout, 1, 1, 1, 1))
        self.kernel_expansion_idx[..., 1] = torch.arange(Cin).reshape((1, 1, Cin, 1, 1))

        self.kernel_expansion_idx2 = self.kernel_expansion_idx.clone()

        idx_r = torch.arange(0, Rin)
        if kernel_type == 'discovered':
            idx_k = torch.Tensor(((0, 0, -1, 0, 0, 0, -1, 0, 0),))

        elif kernel_type == 'baseline':
            idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                                  (4, 3, -1, 5, 0, 2, -1, 6, 1),
                                  (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                  (2, 1, -1, 3, 0, 6, -1, 4, 5),
                                  (1, 6, -1, 2, 0, 5, -1, 3, 4),
                                  (6, 5, -1, 1, 0, 4, -1, 2, 3)))
        else:
            raise ValueError("Invalid kernel type")

        for i in range(self.Rout):
            self.kernel_expansion_idx[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
            self.kernel_expansion_idx[:, i, :, :, :, 3] = idx_k[i,:]

            # our batch norm is in fact invariant to rotations (or whatever the group)
            # so we do not need to roll Rin

    def get_kernel(self):
        idx = self.kernel_expansion_idx
        ret = self.weight[idx[..., 0], idx[..., 1], idx[..., 2], idx[..., 3]]
        ret = ret.reshape((self.Cout, self.Rout, self.Cin, self.Rin, 3, 3))
        ret[..., 0, 2] = 0
        ret[..., 2, 0] = 0
        return ret

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
        if remove_batch_size:
            y = y[0, ...]
        else:
            y = y.reshape(batch_shape + y.shape[1:])

        if self.stride == 2:
            flat_y = y.flatten(0, 2)
            flat_y = torch.nn.functional.pad(flat_y, (0,1,0,1), mode='replicate')
            y = flat_y.unflatten(0, y.shape[:3])
            return self.process_vertices(y)
        else:
            return self.process_vertices(y)


class BatchNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        ret = torch.zeros_like(x)
        x_perm = x.permute(0, 3, 1, 2, 4, 5)
        x_perm = x_perm.mean(dim=3)
        y = self.bn(x_perm.flatten(0, 1))
        y = y.unflatten(1, (-1, 1))
        y = y.unflatten(0, (-1, 5))

        return y.permute(0, 2, 3, 1, 4, 5) 


class GaugeDownLayer(nn.Module):
    def __init__(self, kernel_type, r, c_in, c_out):
        super().__init__()

        self.model = nn.Sequential(
            StrideConv(kernel_type, r, c_in, c_out, 1, stride=2),
            BatchNorm(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class GaugeUpLayer(nn.Module):
    def __init__(self, kernel_type, r, old_c_in, c_in, c_out, activate=True):
        super().__init__()

        self.model = nn.Sequential(
            StrideConv(kernel_type, r + 1, old_c_in + c_in, c_out, 1),
            BatchNorm(c_out),
        )
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
    def __init__(self, kernel_type):
        super().__init__()

        r = ICO_RES

        self.d1 = GaugeDownLayer(kernel_type, r - 0, 16, 16)
        self.d2 = GaugeDownLayer(kernel_type, r - 1, 16, 32)
        self.d3 = GaugeDownLayer(kernel_type, r - 2, 32, 64)
        self.d4 = GaugeDownLayer(kernel_type, r - 3, 64, 128)
        self.d5 = GaugeDownLayer(kernel_type, r - 4, 128, 256)

        self.u5 = GaugeUpLayer(kernel_type, r - 5, 128, 256, 128)
        self.u4 = GaugeUpLayer(kernel_type, r - 4, 64, 128, 64)
        self.u3 = GaugeUpLayer(kernel_type, r - 3, 32, 64, 32)
        self.u2 = GaugeUpLayer(kernel_type, r - 2, 16, 32, 16)
        self.u1 = GaugeUpLayer(kernel_type, r - 1, 0, 16, 3, activate=False)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u5 = self.u5(d4, d5)
        u4 = self.u4(d3, u5)
        u3 = self.u3(d2, u4)
        u2 = self.u2(d1, u3)
        u1 = self.u1(None, u2)

        return torch.sum(u1, dim=-4)


def discover(config):
    train_path = './data/climate'

    print("Task: discovery")

    config.fields = {
        "TMQ": {"mean": 19.21859, "std": 15.81723},
        "U850": {"mean": 1.55302, "std": 8.29764},
        "V850": {"mean": 0.25413, "std": 6.23163},
        "PSL": {"mean": 100814.414, "std": 1461.2227}
    }
    config.label_length = 3  # nothing, AR, TC
    config.field_length = len(config.fields)

    if config.reuse_predictor:
        predictor = torch.load('predictors/climate.pt')
    else:
        predictor = ClimatePredictor(config)
    
    basis = GroupBasis(
        config.field_length, 2, config.label_length, 4, config.standard_basis, 
        lr=5e-4, in_rad=IN_RAD, out_rad=OUT_RAD, 
        identity_in_rep=True,
        identity_out_rep=True, out_interpolation='nearest',
        r3=5.0
    )

    dataset = ClimateTorchDataset(path.join(train_path, 'train'), config)

    gdn = LocalTrainer(ClimateFeatureField, predictor, basis, dataset, config)   
    gdn.train()


def ious(cm):
    i, j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')

    def prune(x):
        if x != x:
            return 0
        return x

    bg_iou = prune(float((cm[0, 0] / torch.sum(cm[(i == 0) | (j == 0)])).detach().cpu()))
    tc_iou = prune(float((cm[1, 1] / torch.sum(cm[(i == 1) | (j == 1)])).detach().cpu()))
    ar_iou = prune(float((cm[2, 2] / torch.sum(cm[(i == 2) | (j == 2)])).detach().cpu()))

    precision = 0
    recall = 0
    for k in range(3):
        precision += prune(float((cm[k,k] / torch.sum(cm[(j == k)])).detach().cpu()))
        recall += prune(float((cm[k,k] / torch.sum(cm[(i == k)])).detach().cpu()))

    return bg_iou, tc_iou, ar_iou, precision/3, recall/3


def dataset_iou(dataset):
    bg_iou = []
    tc_iou = []
    ar_iou = []
    for _, v in dataset.items():
        if len(v) == 1:
            continue

        cm = torch.zeros((3, 3), device=device)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                _, x = torch.max(v[i], dim=0)
                _, y = torch.max(v[j], dim=0)

                for r in range(3):
                    for c in range(3):
                        cm[r, c] += torch.sum((x == r) & (y == c))

        bg, tc, ar = ious(cm)

        bg_iou.append(bg)
        tc_iou.append(tc)
        ar_iou.append(ar)

    bg_iou = np.mean(bg_iou)
    tc_iou = np.mean(tc_iou)
    ar_iou = np.mean(ar_iou)

    iou = np.mean([bg_iou, tc_iou, ar_iou])
    print("dataset ious: bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou)


def train(config, kernel_type):
    print("Task: downstream with kernel type", kernel_type)
    train_path = './data/climate/train'
    test_path = './data/climate/test'

    config.fields = {
        "TMQ": {"mean": 19.2185, "std": 15.8173},
        "U850": {"mean": 1.5530, "std": 8.2976},
        "V850": {"mean": 0.2541, "std": 6.2316},
        "PRECT": {"mean": 2.9458e-08, "std": 1.5564e-07},
        "PSL": {"mean": 100814.0781, "std": 1461.2256},
        "UBOT": {"mean": 0.1249, "std": 6.6533},
        "VBOT": {"mean": 0.3154, "std": 5.7842},
        "QREFHT": {"mean": 0.0078, "std": 0.0062},
        "PS": {"mean": 96571.6172, "std": 9700.1006},
        "T200": {"mean": 213.2091, "std": 7.8898},
        "T500": {"mean": 253.0382, "std": 12.8253},
        "TS": {"mean": 278.7115, "std": 23.6825},
        "TREFHT": {"mean": 278.4212, "std": 22.5119},
        "Z1000": {"mean": 474.1728, "std": 832.8082},
        "Z200": {"mean": 11736.1035, "std": 633.2581},
        "ZBOT": {"mean": 61.3115, "std": 4.9095}
    }
    # from https://github.com/andregraubner/ClimateNet/blob/main/config.json
    config.labels = ["Background", "Tropical Cyclone", "Atmospheric River"]
    config.label_length = 3
    config.field_length = len(config.fields)
    config.lr = 0.001

    train_dataset = ClimateIcoDataset(train_path, config)
    test_dataset = ClimateIcoDataset(test_path, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    date_test_dataset = get_ico_timestamp_dataset(test_dataset)

    model = GaugeEquivariantCNN(kernel_type).to(device)
    print("Model parameter count", sum(p.numel() for p in model.parameters()))

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
            losses.append(float(loss.detach().cpu()))

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        print("Epoch", e, "Loss", np.mean(losses), "IOUs")
        bg_iou, tc_iou, ar_iou, precision, recall = ious(cm)
        iou = torch.tensor([bg_iou, tc_iou, ar_iou]).mean()
        print("bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou)
        print("confusion matrix:\n", cm)

    model.eval()

    bg_iou = []
    tc_iou = []
    ar_iou = []
    pres = []
    recs = []
    for x, ys in tqdm.tqdm(date_test_dataset.values()):
        y_pred = model(x.unsqueeze(0))
        _, y_pred_ind = torch.max(y_pred, dim=1)

        bg = 0
        tc = 0
        ar = 0
        pre = 0
        rec = 0

        for y in ys:
            cm = torch.zeros((3, 3), device=device)
            _, y_true_ind = torch.max(y.unsqueeze(0), dim=1)

            for r in range(3):
                for c in range(3):
                    cm[r, c] += torch.sum((y_true_ind == r) & (y_pred_ind == c))

            bgp, tcp, arp, prp, rep = ious(cm)
            bg += bgp / len(ys)
            tc += tcp / len(ys)
            ar += arp / len(ys)
            pre += prp / len(ys)
            rec += rep / len(ys)
    
        bg_iou.append(bg)
        tc_iou.append(tc)
        ar_iou.append(ar)
        pres.append(pre)
        recs.append(rec)

    bg_iou = np.mean(bg_iou)
    tc_iou = np.mean(tc_iou)
    ar_iou = np.mean(ar_iou)
    pres = np.mean(pres)
    recs = np.mean(recs)
    iou = np.mean([bg_iou, tc_iou, ar_iou])

    torch.set_printoptions(sci_mode=False)
    print("test ious: bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou)
    print("precision:", pres, "recall:", recs)


if __name__ == '__main__':
    c = Config()

    if c.task == 'discover':
        discover(c)
    elif c.task == 'downstream_baseline':
        train(c, 'baseline')
    elif c.task == 'downstream_discovered':
        train(c, 'discovered')
    else:
        print("Unknown task for climate")

