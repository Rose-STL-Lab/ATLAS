import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import SingletonFFTransformer
from config import Config
from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from ff import R2FeatureField
from os import path
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
import numpy as np

device = get_device()

IN_RAD = 350
OUT_RAD = 200

class ClimatePredictor(Predictor):
    def __init__(self, config):
        super().__init__()
        self.network = CGNetModule(classes=config.label_length, channels=config.field_length).to(device)
        self.optimizer = Adam(self.network.parameters(), lr=1e-3)   

    def run(self, x):
        ret = self.network(torch.flatten(x, 0, 1)).unflatten(0, x.shape[:2])

        # clip to the out radius in the case that it's smaller than in radius
        mid_r = x.shape[-2] // 2
        mid_c = x.shape[-1] // 2
        ret = ret[:, :, :, mid_r - OUT_RAD : mid_r + OUT_RAD + 1, mid_c - OUT_RAD : mid_c + OUT_RAD + 1]

        return ret
    
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

if __name__ == '__main__':
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

    gdn = LocalTrainer(R2FeatureField, predictor, basis, dataset, config)   
    gdn.train()
