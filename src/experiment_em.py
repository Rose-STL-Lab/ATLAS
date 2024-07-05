import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, FFConfig
from ff_transformer import R4BilinearFFTransformer

device = get_device()

DIM_SIZE = 12
KEY = 4

def em(scalar_vector):
    # we generally do not really care about the units of these fields
    # as we are primarily focused on the function, and not whether or not
    # the given values are 'physically attainable'
    #
    # return value is same size as input except 6 in final dimension, and x,y,z,w reduced by one


    # padding not supported for this size?
    grad_x = scalar_vector[..., 1:, :, :, :, :] - scalar_vector[..., :-1, :, :, :, :]
    grad_x = torch.cat((grad_x, grad_x[..., -1:, :, :, :, :]), dim=-5)
    grad_y = scalar_vector[..., :, 1:, :, :, :] - scalar_vector[..., :, :-1, :, :, :]
    grad_y = torch.cat((grad_y, grad_y[..., :, -1:, :, :, :]), dim=-4)
    grad_z = scalar_vector[..., :, :, 1:, :, :] - scalar_vector[..., :, :, -1:, :, :]
    grad_z = torch.cat((grad_z, grad_z[..., :, :, -1:, :, :]), dim=-3)
    grad_t = scalar_vector[..., :, :, :, 1:, :] - scalar_vector[..., :, :, :, -1:, :]
    grad_t = torch.cat((grad_t, grad_t[..., :, :, :, -1:, :]), dim=-2)

    e = torch.stack([grad_x[..., 0], grad_y[..., 0], grad_z[..., 0]], dim=-1) - grad_t[..., 1:]
    i_hat = grad_y[..., 3] - grad_z[..., 2]
    j_hat = grad_z[..., 1] - grad_x[..., 3]
    k_hat = grad_x[..., 2] - grad_y[..., 1]
    m = torch.stack([i_hat, j_hat, k_hat], dim=-1)

    full = torch.cat((e, m), dim=-1)
    return full

class EmPredictor(Predictor):
    def run(self, x):
        return em(x)

    def needs_training(self):
        return False


class EmDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        # pow 2
        assert N & (N - 1) == 0
        self.N = N

        # smooth four dimensional function
        interpolator = R4BilinearFFTransformer(DIM_SIZE, KEY)
        sv_key_points = torch.normal(0, 1, (N, KEY ** 4, 4)).to(device)

        with torch.no_grad():
            self.xx = interpolator.smooth_function(sv_key_points)
            self.yy = em(self.xx)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.xx[index], self.yy[index]


if __name__ == '__main__':
    epochs = 25
    N = 1024
    bs = 64

    predictor = EmPredictor()
    transformer = R4BilinearFFTransformer(DIM_SIZE, 5)
    basis = GroupBasis([
        FFConfig('jacobian', 1, lambda_dim=1, manifold_dim=4), 
        FFConfig('jacobian', 3, lambda_dim=1, manifold_dim=4)
    ], transformer, 1)

    dataset = EmDataset(N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)
    # output manifold are very small and thus this needs to be strengthened
    gdn.invariance_fac = 30
    gdn.train(loader, epochs)
