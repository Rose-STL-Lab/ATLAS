import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import TorusFFTransformer

device = get_device()

DIM_SIZE = 12

class LnPredictor(Predictor):
    def run(self, x):
        # integral sgn(v) ln|x(v)| dv where sgn(x) = 1 for top
        # hemisphere, and -1 for bottom hemisphere
        log = torch.log(torch.abs(x))
        top = torch.sum(log[..., DIM_SIZE // 2:], dim=[-2, -1])
        bot = torch.sum(log[..., :DIM_SIZE // 2], dim=[-2, -1])

        return top - bot

    def needs_training(self):
        return False


class LnDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        self.N = N

        self.tensor = torch.normal(0, 1, (N, DIM_SIZE, DIM_SIZE, 1), dtype=torch.complex64).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], LnPredictor().run((self.tensor[index]))


if __name__ == '__main__':
    epochs = 25
    N = 10000
    bs = 64

    predictor = LnPredictor()
    transformer = TorusFFTransformer(DIM_SIZE, DIM_SIZE, 4, 4)
    basis = GroupBasis(1, transformer, 5, 3, dtype=torch.complex64)

    dataset = LnDataset(N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)
