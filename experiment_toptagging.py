# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import argparse, json, time
import pandas as pd
import numpy as np
from utils import get_device, sum_reduce
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, TrivialHomomorphism
from ff_transformer import SingletonFFTransformer
from config import Config
from gnn import LorentzNet, psi
from top import dataset


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

device = get_device()
dtype = torch.float32
    
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
    
    def __call__(self, x):
        return self.run(x)

    def run(self, x):
        # unsqueeze is bc homomorphism assumes at least one manifold dimension
        return self.model(x.reshape(-1, self.n_dim * self.n_components)).unsqueeze(-1)

    def loss(self, y_pred, y_true):
        return nn.functional.cross_entropy(y_pred.squeeze(-1), y_true.squeeze(-1))

    def needs_training(self):
        return True

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=2, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]
        
        self.y = torch.LongTensor(df[:, -1]).unsqueeze(-1).to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run(e, loader, partition):
    tik = time.time()
    loader_length = len(loader)

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    for i, data in enumerate(dataloaders['train']):
        batch_size, n_nodes, _ = data['Pmu'].size()

        if partition == "predictor":
            atom_positions = data['Pmu'].reshape(batch_size * n_nodes, -1).to(device, dtype)
        else:
            xx = data['Pmu'].to(device, dtype)
            yy = data['is_signal'].reshape(-1, 1).to(device, dtype).long()

            xp, yp = basis.apply(xx, yy)
            yp = yp.view(-1).to(device)

            atom_positions = xp.reshape(batch_size * n_nodes, -1).to(device, dtype)

        atom_mask = data['atom_mask'].reshape(batch_size * n_nodes, -1).to(device)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(device)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(device,dtype)
        nodes = psi(nodes)
        edges = [a.to(device) for a in data['edges']]
        label = data['is_signal'].to(device, dtype).long()
        
        pred = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                        edge_mask=edge_mask, n_nodes=n_nodes)
    
        predict = pred.max(1).indices
        
        if partition == "predictor":
            correct = torch.sum(predict == label).item()
            loss = nn.functional.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            correct = torch.sum(predict == yp).item()
            loss = basis.loss(pred, yp) 

            reg = basis.regularization(e)
            loss += reg

            basis.optimizer.zero_grad()
            loss.backward()
            basis.optimizer.step()

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)

        running_loss = sum(res['loss_arr'][-100:])/len(res['loss_arr'][-100:])
        running_acc = sum(res['correct_arr'][-100:])/(len(res['correct_arr'][-100:])*batch_size)
        avg_time = res['time']/res['counter'] * batch_size
        tmp_counter = sum_reduce(res['counter'], device = device)
        tmp_loss = sum_reduce(res['loss'], device = device) / tmp_counter
        tmp_acc = sum_reduce(res['correct'], device = device) / tmp_counter
        
        if i % config.log_interval == 0:
            if partition == "predictor":
                print(">> Predictor: \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                    (e + 1, config.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))
            else:
                print(">> Basis: \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                    (e + 1, config.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))

    res['counter'] = sum_reduce(res['counter'], device = device).item()
    res['loss'] = sum_reduce(res['loss'], device = device).item() / res['counter']
    res['acc'] = sum_reduce(res['correct'], device = device).item() / res['counter']
    print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (res['time'],res['loss'],res['acc']))

    if partition == "predictor":
        torch.save(ddp_model.state_dict(), f"./models/predictor/checkpoint-epoch-{e}.pt")
    
    return 

def train():
    for e in range(0, config.epochs):
        #train predictor
        run(e, dataloaders['train'], "predictor")

        #train basis
        run(e, dataloaders['train'], "basis")

        print("Discovered Basis \n", basis.summary())

if __name__ == '__main__':
    n_dim = 4
    n_class = 2

    config = Config()

    dist.init_process_group(backend='nccl') 
    train_sampler, dataloaders = dataset.retrieve_dataloaders(
                                    config.batch_size,
                                    1,
                                    num_train=-1,
                                    datadir="./data/top-tagging-converted",
                                    nobj = config.n_component)

    predictor = LorentzNet(n_scalar = 1, n_hidden = config.n_hidden, n_class = n_class,
                       dropout = config.dropout, n_layers = config.n_layers,
                       c_weight = config.c_weight)
    predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
    predictor = predictor.to(device)
    ddp_model = DistributedDataParallel(predictor, device_ids=[0])

    ## load best model if needed
    best_model = torch.load(f"./models/predictor/checkpoint-epoch-0.pt")
    ddp_model.load_state_dict(best_model)

    ## predictor optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=0.0003, weight_decay=config.weight_decay)
    
    transformer = SingletonFFTransformer((config.n_component, ))
    homomorphism = TrivialHomomorphism([1], 1)
    basis = GroupBasis(4, transformer, homomorphism, 7, config.standard_basis, loss_type='cross_entropy')

    train()