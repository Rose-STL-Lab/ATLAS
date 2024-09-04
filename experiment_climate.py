import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset, get_timestamp_dataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
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
ICO_RES = 7

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

        return x_ico.unsqueeze(-4), y_ico


# adapted from https://github.com/DavidDiazGuerra/icoCNN/blob/master/icoCNN/icoCNN.py
GL_ORIENTATIONS = 7
class FlatConv(nn.Module):
    def __init__(self, r, Cin, Cout, Rin, Rout=GL_ORIENTATIONS, bias=True, smooth_vertices=False):
        super().__init__()
        assert Rin == 1 or Rin == GL_ORIENTATIONS
        self.r = r
        self.Cin = Cin
        self.Cout = Cout
        self.Rin = Rin
        self.Rout = Rout

        self.process_vertices = SmoothVertices(r) if smooth_vertices else CleanVertices(r)
        self.padding = FlatPadIco(r, Rin, smooth_vertices=smooth_vertices)

        s = math.sqrt(2 / (3 * 3 * Cin * Rin))
        self.weight = torch.nn.Parameter(s * torch.randn((Cout, Cin, Rin, 7)))  # s * torch.randn((Cout, Cin, Rin, 7))  #
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(Cout))
        else:
            self.register_parameter('bias', None)

        self.kernel_expansion_idx = torch.zeros((Cout, Rout, Cin, Rin, 9, 4), dtype=int)
        self.kernel_expansion_idx[..., 0] = torch.arange(Cout).reshape((Cout, 1, 1, 1, 1))
        self.kernel_expansion_idx[..., 1] = torch.arange(Cin).reshape((1, 1, Cin, 1, 1))
        idx_r = torch.arange(0, Rin)
        idx_k = torch.Tensor(((5, 4, -1, 6, 0, 3, -1, 1, 2),
                              (4, 3, -1, 5, 0, 2, -1, 6, 1),
                              (3, 2, -1, 4, 0, 1, -1, 5, 6),
                              (2, 1, -1, 3, 0, 6, -1, 4, 5),
                              (1, 6, -1, 2, 0, 5, -1, 3, 4),
                              (6, 5, -1, 1, 0, 4, -1, 2, 3),
                              (0, 0, -1, 0, 0, 0, -1, 0, 0)))
        for i in range(Rout):
            self.kernel_expansion_idx[:, i, :, :, :, 2] = idx_r.reshape((1, 1, Rin, 1))
            self.kernel_expansion_idx[:, i, :, :, :, 3] = idx_k[i,:]
            idx_r = idx_r.roll(1)

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
        bias = einops.repeat(self.bias, 'Cout -> (Cout Rout)', Cout=self.Cout, Rout=self.Rout) \
            if self.bias is not None else None

        y = torch.nn.functional.conv2d(x, kernel, bias, padding=(1, 1))
        y = einops.rearrange(y, '... (C R) (charts H) W -> ... C R charts H W', C=self.Cout, R=self.Rout, charts=5)
        y = y[..., 1:-1, 1:-1]
        if remove_batch_size: y = y[0, ...]
        else: y = y.reshape(batch_shape + y.shape[1:])

        return self.process_vertices(y)

# remove assertion of r == 1 or r == 6 (don't believe it's even needed)
# this does mean we have to redefine three modules though
class FlatPadIco(torch.nn.Module):
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

class FlatPoolIco(torch.nn.Module):
	def __init__(self, r, R, function=torch.mean, smooth_vertices=False):
		super().__init__()
		self.function = function
		self.padding = FlatPadIco(r, R, smooth_vertices=smooth_vertices)
		self.process_vertices = SmoothVertices(r-1) if smooth_vertices else CleanVertices(r-1)

		self.neighbors = torch.zeros((2**(r-1), 2**r, 7, 2), dtype=torch.long)
		for h in range(self.neighbors.shape[0]):
			for w in range(self.neighbors.shape[1]):
				self.neighbors[h,w,...] = torch.Tensor([[1+2*h,   1+2*w  ],
														[1+2*h+1, 1+2*w  ],
														[1+2*h+1, 1+2*w+1],
														[1+2*h,   1+2*w+1],
														[1+2*h-1, 1+2*w  ],
														[1+2*h-1, 1+2*w-1],
														[1+2*h,   1+2*w-1]])

	def forward(self, x):
		x = self.padding(x)
		receptive_field = x[..., self.neighbors[...,0], self.neighbors[...,1]]
		y = self.function(receptive_field, -1)
		return self.process_vertices(y)

class FlatUnPoolIco(torch.nn.Module):
	def __init__(self, r, R):
		super().__init__()
		self.r = r
		self.R = R
		self.rows = 1+2*torch.arange(2**(r)).unsqueeze(1) # x coord of the center of the hexagonal cell in the unpooled map
		self.cols = 1+2*torch.arange(2**(r+1)).unsqueeze(0) # y coord of the center of the hexagonal cell in the unpooled map
		self.padding = FlatPadIco(r+1, R)

	def forward(self, x):
		y = torch.zeros((x.shape[:-2] + (int(2**(self.r+1)), int(2**(self.r+2)))), device=x.device)
		y = self.padding(y)
		y[..., self.rows, self.cols] = x
		y = y[..., 1:-1, 1:-1]
		return y

class FlatDownLayer(nn.Module):
    def __init__(self, r, c_in, c_out, r_in):
        super().__init__()

        self.model = nn.Sequential(
            FlatConv(r, c_in, c_out, r_in),
            LNormIco(c_out, GL_ORIENTATIONS),
            FlatPoolIco(r, GL_ORIENTATIONS),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class FlatUpLayer(nn.Module):
    def __init__(self, r, c_in, c_out, r_in):
        super().__init__()

        self.model = nn.Sequential(
            FlatUnPoolIco(r, GL_ORIENTATIONS),
            FlatConv(r + 1, c_in, c_out, r_in),
            LNormIco(c_out, GL_ORIENTATIONS),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)


class GaugeDownLayer(nn.Module):
    def __init__(self, r, c_in, c_out, r_in):
        super().__init__()

        self.model = nn.Sequential(
            ConvIco(r, c_in, c_out, r_in),
            LNormIco(c_out, 6),
            PoolIco(r, 6),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)

class GaugeUpLayer(nn.Module):
    def __init__(self, r, c_in, c_out, r_in):
        super().__init__()

        self.model = nn.Sequential(
            UnPoolIco(r, r_in),
            ConvIco(r + 1, c_in, c_out, r_in),
            LNormIco(c_out, 6),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.model(x)


class GaugeEquivariantCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            GaugeDownLayer(ICO_RES - 0,  4, 16, 1),
            GaugeDownLayer(ICO_RES - 1, 16, 32, 6),
            GaugeDownLayer(ICO_RES - 2, 32, 64, 6),
            GaugeDownLayer(ICO_RES - 3, 64, 64, 6),

            GaugeUpLayer(ICO_RES - 4, 64, 64, 6),
            GaugeUpLayer(ICO_RES - 3, 64, 32, 6),
            GaugeUpLayer(ICO_RES - 2, 32, 16, 6),
            GaugeUpLayer(ICO_RES - 1, 16, 3, 6),
        )

    def forward(self, x):
        uncollapsed = self.model(x)
        # collapse 6 orientations
        return torch.sum(uncollapsed, dim=-4)

class GLEquivariantCNN(nn.Module):
    def __init__(self):
        super().__init__()

        o = GL_ORIENTATIONS
        self.model = nn.Sequential(
            FlatDownLayer(ICO_RES - 0,  4, 16, 1),
            FlatDownLayer(ICO_RES - 1, 16, 32, o),
            FlatDownLayer(ICO_RES - 2, 32, 64, o),
            FlatDownLayer(ICO_RES - 3, 64, 64, o),

            FlatUpLayer(ICO_RES - 4, 64, 64, o),
            FlatUpLayer(ICO_RES - 3, 64, 32, o),
            FlatUpLayer(ICO_RES - 2, 32, 16, o),
            FlatUpLayer(ICO_RES - 1, 16, 3, o),
        )

    def forward(self, x):
        uncollapsed = self.model(x)
        return torch.sum(uncollapsed, dim=-4)

def discover():
    train_path = './data/climate'

    config = Config()
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

def train(equivariant, newIOU):
    print("Using equivariant model:", equivariant)
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
    config.train_batch_size = 4
    config.pred_batch_size = 8

    train_dataset = ClimateIcoDataset(train_path, config)
    test_dataset = ClimateIcoDataset(test_path, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=config.batch_size, shuffle=True)


    """
    from icoCNN.plots import icosahedral_charts
    _, y_ind = torch.max(train_dataset[200][0], dim=0)
    icosahedral_charts(train_dataset[1][0][0, 0])
    import matplotlib
    matplotlib.pyplot.show()
    """

    model = GaugeEquivariantCNN()
    optim = torch.optim.Adam(model.parameters())
    for e in range(config.epochs):
        losses = []
        for xx, y_true in tqdm.tqdm(train_loader):

            y_pred = model(xx)

            loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            losses.append(float(loss.detach().cpu()))

            optim.zero_grad()
            loss.backward()
            optim.step()

        print("Loss", np.mean(losses))

    model.eval()

    date_train_dataset = None
    date_test_dataset = None
    if newIOU:
        date_train_dataset = get_timestamp_dataset(train_dataset)
        date_test_dataset = get_timestamp_dataset(test_dataset)

    model = CGNet(equivariant, device, config)
    model.train(train_dataset, date_train_dataset)
    model.evaluate(test_dataset, date_test_dataset)


if __name__ == '__main__':
    #discover()

    # equivariant = True, newIOU = True
    train(True, True)
