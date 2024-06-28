import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import R3BarycentricFFTransformer

# manifold size along each dimension (i.e. 3 -> (3, 3, 3))
MAN_DIM = 4
VECTOR_DIM = 2

device = get_device()


class NormPredictor(Predictor):
    def run(self, x):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    def needs_training(self):
        return False


class NormDataset(torch.utils.data.Dataset):
    def __init__(self, N): 
        self.N = N
        self.tensor = torch.normal(0, 1, (N, MAN_DIM, MAN_DIM, MAN_DIM, VECTOR_DIM)).to(device)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], NormPredictor().run((self.tensor[index]))


if __name__ == '__main__':
    epochs = 25
    N = 10000
    bs = 64

    predictor = NormPredictor()
    transformer = R3BarycentricFFTransformer((MAN_DIM, MAN_DIM, MAN_DIM), 0)
    basis = GroupBasis(VECTOR_DIM, transformer, 5, 3)

    dataset = NormDataset(N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)   
    gdn.train(loader, epochs)

