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

    train_dataset = ClimateDatasetLabeled(train_path, config)
    test_dataset = ClimateDatasetLabeled(test_path, config)
    
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
    # train(True)
