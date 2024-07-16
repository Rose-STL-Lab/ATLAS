import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis 
from ff_transformer import TorusFFTransformer
from config import Config

device = get_device()

DIM_SIZE = 12
VECTOR_DIM = 2

class LnPredictor(Predictor):
    def run(self, x):
        # integral sgn(v) ln|x(v)|^2 dv where sgn(x) = 1 for top
        # hemisphere, and -1 for bottom hemisphere
        log = torch.log(torch.real(torch.sum(x * torch.conj(x), dim=-1)))
        top = torch.sum(log[..., DIM_SIZE // 2:], dim=[-2, -1])
        bot = torch.sum(log[..., :DIM_SIZE // 2], dim=[-2, -1])

        return (top - bot) / (x.shape[-3] * x.shape[-2]) 

    def needs_training(self):
        return False


class LnDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        self.N = N

        # doesnt need to be smooth in this case
        self.tensor = torch.normal(0, 1, (N, DIM_SIZE, DIM_SIZE, VECTOR_DIM), dtype=torch.complex64).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], LnPredictor().run((self.tensor[index]))


if __name__ == '__main__':
    config = Config()
    predictor = LnPredictor()
    transformer = TorusFFTransformer(DIM_SIZE, DIM_SIZE, 4, 4)
    basis = GroupBasis(VECTOR_DIM, transformer, 5, config.standard_basis, dtype=torch.complex64)

    dataset = LnDataset(config.N)

    gdn = LocalTrainer(predictor, basis, dataset, config)
    gdn.train()
