import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset, get_ico_timestamp_dataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_iou_perClass, get_cm_ico
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
class StrideConv(nn.Module):
    def __init__(self, use_gl, r, Cin, Cout, Rin, bias=True, smooth_vertices=False, stride=1):
        super().__init__()
        assert Rin == 1 or Rin == 6
        self.use_gl = use_gl
        self.r = r
        self.Cin = Cin
        self.Cout = Cout
        self.Rin = Rin
        self.stride = stride

        if use_gl:
            self.Rout = 1
        else:
            self.Rout = 6

        rp = r if self.stride == 1 else r - 1
        self.process_vertices = SmoothVertices(rp) if smooth_vertices else CleanVertices(rp)
        self.padding = PadIco(r, Rin, smooth_vertices=smooth_vertices)

        s = math.sqrt(2 / (3 * 3 * Cin * Rin))
        if use_gl:
            self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, 7)))
        else:
            self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, 7)))  # s * torch.randn((Cout, Cin, Rin, 7))  #
        if bias:
            if use_gl:
                self.bias = torch.nn.Parameter(torch.zeros(Cout * self.Rout))
            else:
                self.bias = torch.nn.Parameter(torch.zeros(Cout))
        else:
            self.register_parameter('bias', None)

        self.kernel_expansion_idx = torch.zeros((Cout, self.Rout, Cin, Rin, 9, 4), dtype=int)
        self.kernel_expansion_idx[..., 0] = torch.arange(Cout).reshape((Cout, 1, 1, 1, 1))
        self.kernel_expansion_idx[..., 1] = torch.arange(Cin).reshape((1, 1, Cin, 1, 1))
        idx_r = torch.arange(0, Rin)
        if use_gl:
            idx_k = torch.Tensor(((0, 0, -1, 0, 6, 0, -1, 0, 0),
                                  (1, 1, -1, 1, 6, 1, -1, 1, 1),
                                  (2, 2, -1, 2, 6, 2, -1, 2, 2),
                                  (3, 3, -1, 3, 6, 3, -1, 3, 3),
                                  (4, 4, -1, 4, 6, 4, -1, 4, 4),
                                  (5, 5, -1, 5, 6, 5, -1, 5, 5)))
        else:
            idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                                  (4, 3, -1, 5, 0, 2, -1, 6, 1),
                                  (3, 2, -1, 4, 0, 1, -1, 5, 6),
                                  (2, 1, -1, 3, 0, 6, -1, 4, 5),
                                  (1, 6, -1, 2, 0, 5, -1, 3, 4),
                                  (6, 5, -1, 1, 0, 4, -1, 2, 3)))

        for i in range(self.Rout):
            self.kernel_expansion_idx[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
            self.kernel_expansion_idx[:, i, :, :, :, 3] = idx_k[i,:]
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
        return kernel

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
        if self.use_gl:
            bias = self.bias
        else:
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
            StrideConv(use_gl, r, c_in, c_out, min(r_in, 1 if use_gl else 6), stride=2),
            LNormIco(c_out, 1 if use_gl else 6),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class GaugeUpLayer(nn.Module):
    def __init__(self, use_gl, r, old_c_in, c_in, c_out, activate=True):
        super().__init__()

        self.model = nn.Sequential(
            StrideConv(use_gl, r + 1, old_c_in + c_in, c_out, 1 if use_gl else 6),
            LNormIco(c_out, 1 if use_gl else 6),
        )
        self.lnorm = LNormIco(c_out, 1 if use_gl else 6)
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
            ret = torch.nn.functional.tanh(ret)
        return ret


class GaugeEquivariantCNN(nn.Module):
    def __init__(self, use_gl):
        super().__init__()

        r = ICO_RES

        def c(raw):
            if use_gl:
                return int(math.sqrt(6) * raw)
            else:
                return raw

        self.d1 = GaugeDownLayer(use_gl, r - 0,  4, c(16), 1)
        self.d2 = GaugeDownLayer(use_gl, r - 1, c(16), c(32), 6)
        self.d3 = GaugeDownLayer(use_gl, r - 2, c(32), c(64), 6)
        self.d4 = GaugeDownLayer(use_gl, r - 3, c(64), c(128), 6)
        self.d5 = GaugeDownLayer(use_gl, r - 4, c(128), c(256), 6)

        self.u5 = GaugeUpLayer(use_gl, r - 5, c(128), c(256), c(128))
        self.u4 = GaugeUpLayer(use_gl, r - 4, c(64), c(128), c(64))
        self.u3 = GaugeUpLayer(use_gl, r - 3, c(32), c(64), c(32))
        self.u2 = GaugeUpLayer(use_gl, r - 2, c(16), c(32), c(16))
        self.u1 = GaugeUpLayer(use_gl, r - 1, 0, c(16), 3, activate=False)


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

def train(use_gl, newIOU):
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
    config.lr = 0.001

    train_dataset = ClimateIcoDataset(train_path, config)
    test_dataset = ClimateIcoDataset(test_path, config)

    date_train_dataset = get_ico_timestamp_dataset(train_dataset)
    date_test_dataset = get_ico_timestamp_dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    def print_iou(cm):
        i, j = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        bg_iou = float(cm[0, 0] / torch.sum(cm[(i == 0) | (j == 0)]).detach().cpu())
        tc_iou = float(cm[1, 1] / torch.sum(cm[(i == 1) | (j == 1)]).detach().cpu())
        ar_iou = float(cm[2, 2] / torch.sum(cm[(i == 2) | (j == 2)]).detach().cpu())
        iou = torch.tensor([bg_iou, tc_iou, ar_iou]).mean()
        print("bg", bg_iou, "tc", tc_iou, "ar", ar_iou, "mean", iou)
        print("confusion matrix", cm)

    model = GaugeEquivariantCNN(use_gl).to(device)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.Adam(model.parameters())
    for e in range(config.epochs):
        losses = []

        cm = torch.zeros((3, 3), device=device)
        count = 0
        for xx, y_true, timestamps in tqdm.tqdm(train_loader):
            _, y_true_ind = torch.max(y_true, dim=1)

            y_pred = model(xx)

            _, y_pred_ind = torch.max(y_pred, dim=1)

            if newIOU:
                cm += get_cm_ico(y_pred, 3, timestamps, date_train_dataset, device)
            else:
                for r in range(3):
                    for c in range(3):
                        cm[r, c] += torch.sum((y_true_ind == r) & (y_pred_ind == c))
                count += y_true_ind.numel()

            loss = jaccard_loss(y_pred.flatten(2, 3).cpu(), y_true_ind.flatten(1, 2).cpu())
            # loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            losses.append(float(loss.detach().cpu()))


            optim.zero_grad()
            loss.backward()
            optim.step()

        if newIOU:
            print("Epoch", e, "Loss", np.mean(losses))
            print(cm)
            ious = get_iou_perClass(cm)
            print('IOUs: ', ious, ', mean: ', ious.mean())
        else:
            print("Epoch", e, "Loss", np.mean(losses), "IOUs")
            print_iou(cm / count)


    from icoCNN.plots import icosahedral_charts
    import matplotlib
    icosahedral_charts(y_true_ind[0])
    matplotlib.pyplot.show()
    icosahedral_charts(y_pred_ind[0])
    matplotlib.pyplot.show()

    model.eval()

    print("Test IOU")
    cm = torch.zeros((3, 3), device=device)
    count = 0
    for xx, y_true, timestamps in tqdm.tqdm(test_loader):
        y_pred = model(xx)

        _, y_true_ind = torch.max(y_true, dim=1)
        _, y_pred_ind = torch.max(y_pred, dim=1)

        if newIOU:
            cm += get_cm_ico(y_pred, 3, timestamps, date_test_dataset, device)
        else:
            for r in range(3):
                for c in range(3):
                    cm[r, c] += torch.sum((y_true_ind == r) & (y_pred_ind == c))
            count += y_true_ind.numel()

    if newIOU:
        print(cm)
        ious = get_iou_perClass(cm)
        print('IOUs: ', ious, ', mean: ', ious.mean())
    else:
        print_iou(cm / count)

    """
    date_train_dataset = None
    date_test_dataset = None
    if newIOU:
        date_train_dataset = get_timestamp_dataset(train_dataset)
        date_test_dataset = get_timestamp_dataset(test_dataset)

    model = CGNet(equivariant, device, config)
    model.train(train_dataset, date_train_dataset)
    model.evaluate(test_dataset, date_test_dataset)
    """


if __name__ == '__main__':
    train_path = './data/climate/train'
    test_path = './data/climate/test'

    config = Config()
    config.fields = {"TMQ": {"mean": 19.21859, "std": 15.81723}, 
                     "U850": {"mean": 1.55302, "std": 8.29764},
                     "V850": {"mean": 0.25413, "std": 6.23163},
                     "PSL": {"mean": 100814.414, "std": 1461.2227} 
                    }

    #discover()

    # use_gl = True, newIOU = True
    train(False, True)

