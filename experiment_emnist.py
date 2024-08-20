import torch
import torch.nn as nn
import escnn
import escnn.nn as enn
from utils import get_device, ManifoldPredictor
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff import R2FeatureField
from config import Config
from torchvision import datasets, transforms

IN_RAD = 14
OUT_RAD = 14
NUM_CLASS = 62

device = get_device()


# predicts on regions
class EMNISTPredictor(nn.Module, Predictor):
    def __init__(self):
        super(EMNISTPredictor, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        ).to(device)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 62, kernel_size=3, padding=1),  
            nn.LeakyReLU(),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def run(self, x):
        org_batch = x.shape[:2]
        batched = torch.flatten(x, end_dim=1)

        enc = self.down(batched)
        dec = self.up(enc)
        ret = dec.unflatten(0, org_batch)
        return ret

    def loss(self, xx, yy):
        xx = xx.permute(0, 1, 3, 4, 2).flatten(0, 3)
        yy = yy.permute(0, 1, 3, 4, 2).flatten(0, 3)
        return nn.functional.cross_entropy(xx, yy)

    def needs_training(self):
        return False


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.EMNIST(
            root='./data',
            split='balanced',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        y_onehot = torch.zeros(62, device=device)
        y_onehot[y] = 1

        y_expanded = y_onehot.unsqueeze(1).unsqueeze(2).expand(-1, 28, 28)

        return x.to(device), y_expanded

def discover():
    config = Config()

    # predictor = EMNISTPredictor()
    predictor = torch.load('predictor.pt')
    
    basis = GroupBasis(
        1, 2, 62, 1, config.standard_basis, 
        loss_type='cross_entropy', lr=5e-4, in_rad=IN_RAD, out_rad=OUT_RAD, 
        identity_out_rep=True, # matrix exp of 62 x 62 matrix generally becomes nan
    )

    dataset = EMNISTDataset(config.N)

    gdn = LocalTrainer(R2FeatureField, predictor, basis, dataset, config)   
    gdn.train()

def train(act):
    import tqdm 

    config = Config()

    dataset = EMNISTDataset(config.N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ManifoldPredictor(act, [
            1  * [act.trivial_repr],
            64 * [act.regular_repr],
            64 * [act.regular_repr],
            64 * [act.regular_repr],
            62 * [act.trivial_repr],
        ], R2FeatureField)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(config.epochs):
        total_loss = 0
        for xx, yy in tqdm.tqdm(loader):
            y_pred = model(xx)

            y_pred = y_pred.permute(0, 2, 3, 1).flatten(0, 2)
            yy = yy.permute(0, 2, 3, 1).flatten(0, 2)
            loss = torch.nn.functional.cross_entropy(y_pred, yy)

            total_loss += loss / len(loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss", total_loss)


if __name__ == '__main__':
    discover()
    # train(escnn.gspaces.trivialOnR2())
    # train(escnn.gspaces.rot2dOnR2(8))
