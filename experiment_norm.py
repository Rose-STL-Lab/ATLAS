import torch
import torch.nn as nn
from ..utils import get_device
from ..local_symmetry import Predictor, LocalTrainer
from ..group_basis import GroupBasis
from ..ff_transformer import TorusFFTransformer
from ..config import Config

# torus size along each dimension (i.e. 3 -> (3, 3, 3))
MAN_DIM = 10
VECTOR_DIM = 3

device = get_device()


class NormPredictor(Predictor):
    def run(self, x):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    def needs_training(self):
        return False


class NormDataset(torch.utils.data.Dataset):
    def __init__(self, N): 
        self.N = N
        self.tensor = torch.normal(0, 1, (N, MAN_DIM, MAN_DIM, VECTOR_DIM)).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], NormPredictor().run((self.tensor[index]))


if __name__ == '__main__':
    epochs = 25
    N = 10000
    bs = 64

    config = Config()

    predictor = NormPredictor()
    
    # the sub division rate doesn't really matter as we don't require smoothness anyway
    transformer = TorusFFTransformer(MAN_DIM, MAN_DIM, 5, 5)
    basis = GroupBasis(VECTOR_DIM, transformer, 3, config.standard_basis, lr=5e-4)

    dataset = NormDataset(N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)   
    gdn.train(loader, epochs)

