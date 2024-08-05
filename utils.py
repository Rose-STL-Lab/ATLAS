import torch
import numpy as np
import torch.distributed as dist

def rmse(xx, yy):
    return torch.sqrt(torch.mean(torch.square(xx - yy)))


def mae(xx, yy):
    return torch.mean(torch.abs(xx - yy))


def get_device(no_mps=True):
    if no_mps:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def volume_tetrahedron(p, q, r, s):
    return 1 / 6 * np.abs(np.linalg.det([
        [*p, 1], 
        [*q, 1],
        [*r, 1],
        [*s, 1]
    ]))


def tetrahedron_contains(p, q, r, s, test):
    total = volume_tetrahedron(p, q, r, s)
    p_prime = volume_tetrahedron(test, q, r, s)
    q_prime = volume_tetrahedron(p, test, r, s)
    r_prime = volume_tetrahedron(p, q, test, s)
    s_prime = volume_tetrahedron(p, q, r, test)

    return p_prime + q_prime + r_prime + s_prime <= total + 1e-5


# precondition: test contained within the tetrahedron
def barycentric_3d(p, q, r, s, test):
    total = volume_tetrahedron(p, q, r, s)
    p_prime = volume_tetrahedron(test, q, r, s)
    q_prime = volume_tetrahedron(p, test, r, s)
    r_prime = volume_tetrahedron(p, q, test, s)
    # technically redundant
    s_prime = volume_tetrahedron(p, q, r, test)

    assert p_prime + q_prime + r_prime + s_prime <= total + 1e-5

    return [p_prime / total, q_prime / total, r_prime / total, s_prime / total]


# 'volume'
def volume_triangle(p, q, r):
    qp = p - q
    qr = r - p
    return 0.5 * np.linalg.norm(np.cross(qp, qr))


def barycentric_2d(p, q, r, test):
    total = volume_triangle(p, q, r)
    p_prime = volume_triangle(test, q, r)
    q_prime = volume_triangle(p, test, r)
    r_prime = volume_triangle(p, q, test)

    assert p_prime + q_prime + r_prime <= total + 1e-3

    return (p_prime / total, q_prime / total, r_prime / total)

def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor

def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt