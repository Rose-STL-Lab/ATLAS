import torch
import torch.nn as nn
import math
from utils import get_device, ManifoldPredictor
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff import R2FeatureField
from config import Config
import torchvision
from torchvision import datasets, transforms

device = get_device()

IN_RAD = 14
OUT_RAD = 14
CLASS_WEIGHTS = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device)
NUM_CLASS = len(CLASS_WEIGHTS)


class MNISTFeatureField(R2FeatureField):
    def __init__(self, data):
        super().__init__(data)

        w = self.data.shape[-1]
        h = self.data.shape[-2]
        locs = [(h / 6, w * 0.5), (h * 0.5, w * 0.5), (h * 5 / 6, w * 0.5)]

        self.locs = [(int(r), int(c)) for r, c in locs]
    
    def regions(self, radius):
        """ we assume a priori knowledge of a good chart
        In this case, it makes sense to do equirectangular projection
        (especially since that's used in the projection code),
        but theoretically similar projections should work as well"""

        max_r = self.data.shape[-2] / (2 * math.pi)
        assert radius < max_r

        ret = torch.zeros(self.data.shape[0], len(self.locs), self.data.shape[1], 2 * radius + 1, 2 * radius + 1)
        inds = torch.arange(-radius, radius + 1, device=device) / max_r
        phi = torch.asin(inds)
        phi_inds = (phi / math.pi * self.data.shape[-1] + self.data.shape[-1] // 2).round().long()

        for i, (p0, _) in enumerate(self.locs):
            ret[:, i] = self.data[
                :, 
                :, 
                torch.arange(p0 - radius, p0 + radius + 1, device=device).unsqueeze(0), 
                phi_inds.unsqueeze(1)
            ]

        return ret.to(device)


# predicts on a single region
class SinglePredictor(nn.Module):
    def __init__(self):
        super().__init__()

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

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, NUM_CLASS, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(NUM_CLASS),
        ).to(device)

    def forward(self, x):
        enc = self.down(x.to(device))
        dec = self.up(enc.to(device))
        return dec[:, :, :29, :29]


class MNISTPredictor(nn.Module, Predictor):
    def __init__(self):
        super(MNISTPredictor, self).__init__()
        
        self.c1 = SinglePredictor()
        self.c2 = SinglePredictor()
        self.c3 = SinglePredictor()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.run(x)

    def run(self, x):
        return torch.stack([
            self.c1(x[:, 0]),
            self.c2(x[:, 1]),
            self.c3(x[:, 2]),
        ], dim=1).to(device)

    def loss(self, xx, yy):
        xx = xx.permute(0, 1, 3, 4, 2).flatten(0, 3)
        yy = yy.permute(0, 1, 3, 4, 2).flatten(0, 3)
        return nn.functional.cross_entropy(xx, yy, weight=CLASS_WEIGHTS).to(device)

    def name(self):
        return "mnist"

    def needs_training(self):
        return True

    def returns_logits(self):
        return True


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, N, rotate=180, train=True):
        self.dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

        self.w = 120
        self.h = self.w // 2

        self.x = torch.zeros(N, 1, self.w, self.h, device=device)
        self.y = torch.zeros(N, NUM_CLASS, self.w, self.h, device=device)

        h = lambda x : hash(str(x))

        for i in range(N):
            j = [h(i) % N, (h(i) + 1) % N, (h(i) + 2) % N]
            starts = [(int(self.w / 6), 16), (int(self.w / 2), 16), (int(self.w * 5 / 6), 16)]

            # x_flat/y_flat represent the digits on a cylinder
            # we then project onto the sphere (equirectangular)
            x_flat = torch.zeros(1, self.w, 32, device=device)
            y_flat = torch.zeros(NUM_CLASS, self.w, 32, device=device)

            for jp, (r, c) in list(zip(j, starts)):
                theta = h(i + jp) % (2 * rotate) - rotate if rotate else 0
                x, y = self.dataset[jp]
                x_curr = torchvision.transforms.functional.rotate(x, theta)
                x_flat[:, r - 14: r + 14, c - 14: c + 14] = x_curr

                p = 32
                y_curr = torch.zeros(NUM_CLASS, p, p)
                y_curr[y] = 1

                y_flat[:, r - p // 2: r + p // 2, c - p // 2: c + p // 2] = y_curr

            # only label pixels with white
            y_flat *= (x_flat[0] != 0).unsqueeze(0)

            self.x[i] = self.project(x_flat)
            self.y[i] = self.project(y_flat)
            # label unmarked pixels as background
            self.y[i][10] = torch.sum(self.y[i], dim=-3) == 0

    # equirectangular nearest neighbor
    def project(self, cylinder):
        ret = torch.zeros((cylinder.shape[0], self.w, self.h), device=device)         

        inds = torch.arange(-self.h // 2, self.h // 2, device=device)
        phi = inds * math.pi / self.h

        r = self.w / (2 * math.pi)
        y_val = (torch.sin(phi) * r + cylinder.shape[-1] / 2).round().long()
        
        mask = (y_val >= 0) & (y_val < cylinder.shape[-1])
        i_val = torch.arange(0, self.h, device=device)[mask]
        y_val = y_val[mask]

        x = torch.arange(self.w).unsqueeze(1)
        ret[:, x, i_val.unsqueeze(0)] = cylinder[:, x, y_val.unsqueeze(0)]

        return ret

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def discover(config):
    print("Task: discovery")

    if config.reuse_predictor:
        predictor = torch.load('predictors/mnist.pt')
    else:
        predictor = MNISTPredictor()

    basis = GroupBasis(
        1, 2, NUM_CLASS, 1, config.standard_basis, 
        lr=5e-4, in_rad=IN_RAD, out_rad=OUT_RAD, 
        identity_in_rep=True, identity_out_rep=True
    )

    dataset = MNISTDataset(config.N, rotate=60)

    gdn = LocalTrainer(MNISTFeatureField, predictor, basis, dataset, config)   
    gdn.train()


debug = 0
def lie_gan_discover(config):
    """
        In general, since the y labels are at fixed positions on the sphere, we do not expect 
        LieGan to be able to discover any (continuous) symmetries, since none exist
    """
    from SO3LieGan.gan import LieGenerator, LieDiscriminatorSegmentation
    from SO3LieGan.train import train_lie_gan

    print("Task: discovery with LieGAN")

    so3_basis = torch.tensor([
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[0, 0, -1],
         [0, 0, 0],
         [1, 0, 0]],
        [[0, 0, 0],
         [0, 0, -1],
         [0, 1, 0]]
    ], device=device)

    # transforms field by a SO3 rotation
    def transform(g, x_in, is_y):
        # theta phi for each point
        max_theta = x_in.shape[-2]
        max_phi = x_in.shape[-1]
        theta = torch.arange(0, max_theta) / max_theta * 2 * math.pi
        phi = torch.arange(0, max_phi) / max_phi * math.pi
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')

        # x y z for each pixel
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack((x, y, z), dim=-1).to(device)

        # inverted x y z for each pixel
        xyz_inv = torch.einsum('bvw, ijw -> bijv', torch.inverse(g), xyz)

        # mapped theta phi for each pixel
        tau = 2 * math.pi
        theta_inv = (torch.atan2(xyz_inv[..., 1], xyz_inv[..., 0]) + tau) % tau / tau
        theta_inv[theta_inv.isnan()] = 0
        theta_inv = 2 * theta_inv - 1

        # floating point error
        phi_inv = (torch.acos(xyz_inv[..., 2].clamp(-0.9999, 0.9999)) / math.pi) 
        phi_inv = 2 * phi_inv - 1
        theta_phi_inv = torch.stack((phi_inv, theta_inv), dim=-1)

        # sampled 
        if is_y:
            ret = torch.nn.functional.grid_sample(x_in, theta_phi_inv, mode='nearest', align_corners=False) 
        else:
            ret = torch.nn.functional.grid_sample(x_in, theta_phi_inv, mode='bilinear', align_corners=False) 

        global debug
        debug += 1
        if debug == -1:
            import matplotlib.pyplot as plt
            # , y_ind = torch.max(ret[0], 
            plt.imshow(yind.swapaxes(0, 2).cpu().detach())
            plt.show()
            plt.imshow(x_in[0].swapaxes(0, 2).cpu().detach())
            plt.show()
        return ret

    generator = LieGenerator(1, transform, so3_basis).to(device)
    discriminator = LieDiscriminatorSegmentation(1, 768, NUM_CLASS).to(device)

    dataset = MNISTDataset(config.N, rotate = 60)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    train_lie_gan(generator, discriminator, loader, config.epochs, 2e-4, 1e-3, 'cosine', 1e-2, 2, 0.0, 1.0, device, print_every=1)


def train(G, config):
    import tqdm

    print("Task: downstream MNIST training with group =", G)

    dataset = MNISTDataset(config.N, rotate=60)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    valid_dataset = MNISTDataset(10000, train=False, rotate=180)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)

    model = ManifoldPredictor([
        [1, 32, 1],
        [32, 32, 1],
        [32, 64, 1],
        [64, 64, 1],
        [64, 64, 2],
        [64, 32, 2],
        [32, 32, 2],
        [32, 32, 2],
        [32, 32, 1],
        [32, 16, 1],
        [16, 16, 1],
        [16, NUM_CLASS, 1],
    ], MNISTFeatureField, G)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # we basically do not care about background
    # this does mean that some background pixels will be marked somewhat randomly
    # but that's generally not really an issue
    # as really we treat this task more as classification than explcit segmentation
    # the segmentation part is of course very easy (just white vs non white)
    cw = torch.tensor(10 * [1] + [0.005], device=device)

    for e in range(config.epochs):
        total_loss = 0
        train_acc = 0

        model.train()
        for xx, yy in tqdm.tqdm(loader):
            y_pred = model(xx)

            y_pred = y_pred.permute(0, 2, 3, 1).flatten(0, 2)
            yy = yy.permute(0, 2, 3, 1).flatten(0, 2)

            loss = torch.nn.functional.cross_entropy(y_pred, yy, weight=cw)

            y_pred_ind = torch.max(y_pred, dim=-1, keepdim=True)[1]
            y_true_ind = torch.max(yy, dim=-1, keepdim=True)[1]
            non_bg = y_true_ind != 10
            train_acc += (y_pred_ind[non_bg] == y_true_ind[non_bg]).float().mean() / len(loader)

            total_loss += loss / len(loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, 'predictors/mnist_downstream_' + G + '.pt')
        print("Loss", total_loss, "Train Accuracy", train_acc)

    total_acc = 0
    model.eval()
    for xx, yy in tqdm.tqdm(valid_loader):
        y_pred = model(xx)

        y_pred = y_pred.permute(0, 2, 3, 1).flatten(0, 2)
        yy = yy.permute(0, 2, 3, 1).flatten(0, 2)

        y_pred_ind = torch.max(y_pred, dim=-1)[1]
        y_true_ind = torch.max(yy, dim=-1)[1]

        # we do not include background pixels
        nonbg = y_true_ind != 10

        total_acc += (y_pred_ind[nonbg] == y_true_ind[nonbg]).float().mean() * len(xx) / len(valid_dataset)

    print("Test Accuracy", total_acc)


if __name__ == '__main__':
    c = Config()

    if c.task == 'discover':
        discover(c)
    elif c.task == 'liegan_discover':
        lie_gan_discover(c)
    elif c.task == 'downstream_baseline':
        train('trivial', c)
    elif c.task == 'downstream_discovered':
        train('so2', c)
    else:
        print("Unknown task for MNIST")
