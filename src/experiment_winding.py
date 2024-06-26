import itertools
import math

import torch
import torch.nn as nn
import random
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import TorusFFTransformer

U_SAMPLES = 45
U_KSAMPLES = 5
V_SAMPLES = 45
V_KSAMPLES = 5

assert U_SAMPLES % U_KSAMPLES == 0
assert V_SAMPLES % V_KSAMPLES == 0

device = get_device()

NUM_LOOPS = 1


torch.manual_seed(0)
random.seed(0)
class WindingPredictor(Predictor):
    def __init__(self):
        self.paths = []

        for i in range(NUM_LOOPS):
            u_bias = U_SAMPLES * torch.poisson(torch.scalar_tensor(0.5))
            v_bias = V_SAMPLES * torch.poisson(torch.scalar_tensor(0.5))
            num_down = int(torch.poisson(torch.scalar_tensor(U_SAMPLES / 2)))
            num_up = int(num_down + u_bias)
            num_left = int(torch.poisson(torch.scalar_tensor(V_SAMPLES / 2)))
            num_right = int(num_left + v_bias)

            path_str = []
            for j in range(num_down):
                path_str.append('d')
            for j in range(num_up):
                path_str.append('u')
            for j in range(num_left):
                path_str.append('l')
            for j in range(num_right):
                path_str.append('r')

            random.shuffle(path_str)

            u = random.randint(0, U_SAMPLES - 1)
            v = random.randint(0, V_SAMPLES - 1)
            path = [(u, v)]
            for p in path_str:
                if p == 'd':
                    v -= 1
                elif p == 'u':
                    v += 1
                elif p == 'l':
                    u -= 1
                else:
                    u += 1

                u = (u + U_SAMPLES) % U_SAMPLES
                v = (v + V_SAMPLES) % V_SAMPLES

                path.append((u, v))
            self.paths.append(path)

    def run(self, x):
        # manifold size and vector index will be elided
        ret = torch.zeros((*x.shape[:-3], NUM_LOOPS)).to(device)

        for i, p in enumerate(self.paths):
            for prev, curr in itertools.pairwise(p):
                prev_theta = torch.atan2(x[..., prev[0], prev[1], 1], x[..., prev[0], prev[1], 0])
                curr_theta = torch.atan2(x[..., curr[0], curr[1], 1], x[..., curr[0], curr[1], 0])
                delta_theta = curr_theta - prev_theta
                delta_theta[delta_theta > math.pi] -= 2 * math.pi
                delta_theta[delta_theta < -math.pi] += 2 * math.pi
                ret[..., i] += delta_theta

                # should be smooth
                assert torch.max(torch.abs(delta_theta)) < 2

        return ret

    def needs_training(self):
        return False


class WindingDataset(torch.utils.data.Dataset):
    def __init__(self, N, predictor, gen_upoints=3, gen_vpoints=3):
        self.N = N
        self.predictor = predictor

        smooth_function_gen = TorusFFTransformer(U_SAMPLES, V_SAMPLES, gen_upoints, gen_vpoints)
        key_points = torch.empty((N, gen_upoints * gen_vpoints, 2)).to(device)
        key_points[:, :, 0] = 1 + torch.abs(torch.normal(0, 1, (N, gen_upoints * gen_vpoints))).to(device)
        key_points[:, :, 1] = 2 * math.pi * torch.rand((N, gen_upoints * gen_vpoints)).to(device)
        # interpolate the magnitude and angle
        lerped = smooth_function_gen.smooth_function(key_points)
        self.tensor = torch.stack(
            [torch.cos(lerped[..., 1]) * lerped[..., 0],
             torch.sin(lerped[..., 1]) * lerped[..., 0]],
            dim=3
        )

        self.out_tensor = self.predictor.run(self.tensor)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], self.out_tensor[index]


if __name__ == '__main__':
    epochs = 25
    N = 1000
    bs = 64

    predictor = WindingPredictor()
    transformer = TorusFFTransformer(U_SAMPLES, V_SAMPLES, U_KSAMPLES, V_KSAMPLES)
    basis = GroupBasis(2, transformer, 8, 1)
    basis.predictor = predictor

    dataset = WindingDataset(N, predictor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)

    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)
