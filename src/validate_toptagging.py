import torch
import pandas as pd
from experiment_toptagging import ClassPredictor
from utils import get_device

device = get_device()
model = torch.load('models/toptagclass.pt')

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/val.h5', flatten=False, n_component=30, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = torch.FloatTensor(self.X)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]
        
        self.y = torch.LongTensor(df[:, -1])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataloader = torch.utils.data.DataLoader(TopTagging(), batch_size=64)

delta = 0
for xx, yy in dataloader:
    res = model.run(xx)
    _, ind = torch.max(res, dim=-1)
    yy = yy.to(device)
    delta += (ind == yy).sum() / len(dataloader) / len(xx)

print(delta)
