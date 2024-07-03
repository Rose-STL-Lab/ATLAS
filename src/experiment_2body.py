# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master/baseline/augerino.py)

import torch
import torch.nn as nn
import pickle
import numpy as np
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import R3BarycentricFFTransformer

# manifold size along each dimension (i.e. 3 -> (3, 3, 3))
MAN_DIM = 4
VECTOR_DIM = 2

device = get_device()

class TrajPredictor(Predictor):
    def __init__(self, n_dim, n_input_timesteps, n_output_timesteps):
        self.model = nn.Sequential(
            nn.Linear(n_dim * n_input_timesteps, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_dim * n_output_timesteps),
        )

    def run(self, x):
        return self.model(x.view(-1, self.n_dim * self.n_input_timesteps)).view(-1, self.n_output_timesteps, self.n_dim)

    def needs_training(self):
        return False

class NBodyDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='./data/2body-orbits-dataset.pkl', mode='train', trj_timesteps=50, input_timesteps=4, output_timesteps=1, extra_features=None, flatten=False, with_random_transform=False, nbody=2):
        with open(save_path, 'rb') as f:
            self.data = pickle.load(f)
        if mode == 'train':
            self.data = self.data['coords']
        else:
            self.data = self.data['test_coords']
        self.feat_dim = nbody * 4
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(-1, trj_timesteps, self.feat_dim)
        if nbody == 2:
            self.data = self.data[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
        else:
            raise NotImplementedError

        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.X, self.y = [], []
        self.N = self.data.shape[0]
        trj_timesteps = self.data.shape[1]
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps):
                self.X.append(self.data[i, t:t+input_timesteps, :])
                self.y.append(self.data[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.len = self.X.shape[0]
        if with_random_transform:
            if nbody == 2:
                GT = np.zeros((1, 8, 8))
                GT[0,1,0]=GT[0,3,2]=GT[0,5,4]=GT[0,7,6]=-1
                GT[0,0,1]=GT[0,2,3]=GT[0,4,5]=GT[0,6,7]=1
            elif nbody == 3:
                GT = np.zeros((1, 12, 12))
                GT[0,1,0]=GT[0,3,2]=GT[0,5,4]=GT[0,7,6]=GT[0,9,8]=GT[0,11,10]=-1
                GT[0,0,1]=GT[0,2,3]=GT[0,4,5]=GT[0,6,7]=GT[0,8,9]=GT[0,10,11]=1
            GT = torch.tensor(GT, dtype=torch.float32)
            z = torch.randn(self.X.shape[0], 1)
            g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', GT, z))
            self.gx = torch.einsum('bij,bkj->bki', g_z, self.X)
            self.gy = torch.einsum('bij,bkj->bki', g_z, self.y)
        if flatten:
            self.X = self.X.reshape(self.len, -1)
            self.y = self.y.reshape(self.len, -1)
            if with_random_transform:
                self.gx = self.gx.reshape(self.len, -1)
                self.gy = self.gy.reshape(self.len, -1)
        self.with_random_transform = with_random_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.with_random_transform:
            return self.X[idx], self.y[idx], self.gx[idx], self.gy[idx]
        else:
            return self.X[idx], self.y[idx]


if __name__ == '__main__':
    epochs = 25
    N = 1000
    bs = 64
    input_timesteps = 1
    output_timesteps = 1
    n_channel = 1
    n_component = 1
    n_dim = 8
    
    # predictor from LieGAN codebase (used for augerino)
    predictor = TrajPredictor(n_dim, input_timesteps, output_timesteps)

    # TODO : Need to figure out how to modify group basis for this dataset
    transformer = R3BarycentricFFTransformer((MAN_DIM, MAN_DIM, MAN_DIM), 0)
    basis = GroupBasis(VECTOR_DIM, transformer, 5, 3)
    basis.predictor = predictor

    # dataset from LieGAN codebase 
    dataset = NBodyDataset(
        input_timesteps=input_timesteps,
        output_timesteps=output_timesteps,
        save_path=f'./data/2body-orbits-dataset.pkl',
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    # TODO : Need to figure how to make train work with given data
    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)
