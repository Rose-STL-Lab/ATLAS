import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import itertools
import sys
from utils import *

device = get_device()


# Feature Field Transformer
# Given the group action elements, actually apply a local transformation to
# the target manifold feature field
class FFTransformer(ABC):
    def __init__(self, blend_key, blend_val):
        super().__init__()
        self.blend_key = blend_key
        self.blend_val = blend_val

    def num_key_points(self):
        return torch.max(self.blend_key) + 1

    # value at kp: [bs (exactly one dimension), num_key_points, *shape]
    # ret [bs, *manifold_shape, *shape]
    def smooth_function(self, value_at_key_points):
        blend_v = self.blend_val.unsqueeze(0)
        for i in range(len(value_at_key_points.shape) - 2):
            blend_v = blend_v.unsqueeze(-1)

        manifold_size = self.blend_key.shape[:-1]

        mult = value_at_key_points[:, self.blend_key] * blend_v
        mult = torch.sum(mult, dim=len(manifold_size) + 1)

        return mult

    def apply_lie(self, group_key_points, feature_field):
        """
            group_key_points: elements of the lie algebra of shape [batch, num_key_points, ff_dim, ff_dim]
            feature_field: the actual input value tensor of shape [batch, *manifold_shape, ff_dim]

            A(x) -> A(x) ^ e^{\Lambda(x))}
        """

        mult = self.smooth_function(group_key_points)
        # shape: [(batch), *manifold_size, ff_dimension, ff_dimension]
        matrices = torch.matrix_exp(mult)

        return (matrices @ feature_field.unsqueeze(-1)).squeeze(-1)


class TorusFFTransformer(FFTransformer):
    def __init__(self, u_dim, v_dim, u_keypoints, v_keypoints):
        assert u_dim % u_keypoints == 0
        assert v_dim % v_keypoints == 0

        blend_key = torch.zeros((u_dim, v_dim, 4), dtype=torch.int).to(device)
        blend_val = torch.zeros((u_dim, v_dim, 4)).to(device)

        for u in range(u_dim):
            for v in range(v_dim):
                # bi linear interpolate off of four closest points
                u0 = u * u_keypoints // u_dim
                u1 = (u0 + 1) % u_keypoints
                v0 = v * v_keypoints // v_dim
                v1 = (v0 + 1) % v_keypoints

                s = (u - (u0 * u_dim // u_keypoints)) / (u_dim // u_keypoints)
                t = (v - (v0 * v_dim // v_keypoints)) / (v_dim // v_keypoints)

                blend_key[u, v, 0] = u0 * v_keypoints + v0
                blend_key[u, v, 1] = u1 * v_keypoints + v0
                blend_key[u, v, 2] = u0 * v_keypoints + v1
                blend_key[u, v, 3] = u1 * v_keypoints + v1

                blend_val[u, v, 0] = (1 - s) * (1 - t)
                blend_val[u, v, 1] = s * (1 - t)
                blend_val[u, v, 2] = (1 - s) * t
                blend_val[u, v, 3] = s * t
        
        super().__init__(blend_key, blend_val)

class SingletonFFTransformer(FFTransformer):
    def __init__(self, manifold_size):
        key = torch.zeros((*manifold_size, 1), device=device, dtype=torch.long)
        val = torch.ones((*manifold_size, 1), device=device)
        super().__init__(key, val)

class R1FFTransformer(FFTransformer):
    def __init__(self, dim, kdim):
        assert dim % (kdim - 1) == 0
        blend_key = torch.zeros((dim, 2), dtype=torch.int).to(device)
        blend_val = torch.zeros((dim, 2)).to(device)

        block = dim // (kdim - 1)
        for x in range(dim):
            xind = x // block
            xs = x % block / block 

            for i in [0, 1]:
                xp = xind + i

                prod = xs if i else 1 - xs
                blend_key[x,i] = xp 
                blend_val[x,i] = prod

        super().__init__(blend_key, blend_val)


# not really bilinear, but i dont know generalization of name
class R4BilinearFFTransformer(FFTransformer):
    def __init__(self, dim, kdim):
        assert dim % (kdim - 1) == 0

        blend_key = torch.zeros((dim, dim, dim, dim, 2 ** 4), dtype=torch.int).to(device)
        blend_val = torch.zeros((dim, dim, dim, dim, 2 ** 4)).to(device)

        block = dim // (kdim - 1)

        for (x, y, z, t) in itertools.product(range(dim), repeat=4):
            xind = x // block
            yind = y // block
            zind = z // block
            tind = t // block

            xs = x % block / block 
            ys = y % block / block 
            zs = z % block / block 
            ts = t % block / block 
            s = (xs, ys, zs, ts)

            for i, vertex in enumerate(itertools.product([0, 1], repeat=4)):
                xp = xind + vertex[0]
                yp = yind + vertex[1]
                zp = yind + vertex[2]
                tp = tind + vertex[3]
                prod = 1
                for i in range(4):
                    if vertex[i] == 0:
                        prod *= (1 - s[i])
                    else:
                        prod *= s[i]
                
                blend_key[x,y,z,t,i] = xp * kdim ** 3 + yp * kdim ** 2 + zp * kdim + tp 
                blend_val[x,y,z,t,i] = prod

        super().__init__(blend_key, blend_val)

    def jacobian(self, field):
        grad_x = field[..., 1:, :, :, :, :] - field[..., :-1, :, :, :, :]
        grad_x = torch.cat((grad_x, grad_x[..., -1:, :, :, :, :]), dim=-5)
        grad_y = field[..., :, 1:, :, :, :] - field[..., :, :-1, :, :, :]
        grad_y = torch.cat((grad_y, grad_y[..., :, -1:, :, :, :]), dim=-4)
        grad_z = field[..., :, :, 1:, :, :] - field[..., :, :, -1:, :, :]
        grad_z = torch.cat((grad_z, grad_z[..., :, :, -1:, :, :]), dim=-3)
        grad_t = field[..., :, :, :, 1:, :] - field[..., :, :, :, -1:, :]
        grad_t = torch.cat((grad_t, grad_t[..., :, :, :, -1:, :]), dim=-2)

        return torch.cat((grad_x, grad_y, grad_z, grad_t), dim=-1)
        
"""
def r3_blending_matrix(ff_shape, subdivisions):
    num_segments = 2 ** subdivisions
    key_points = (num_segments + 1) ** 3
    ret = torch.empty((*ff_shape, key_points)).to(device)

    # bitindices of vertices of tetrahedrons that form a cube
    tetrahedron_offsets = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0]],

        [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
    ])

    for x in range(ff_shape[0]):
        for y in range(ff_shape[1]):
            for z in range(ff_shape[2]):
                coord = np.array([x / ff_shape[0], y / ff_shape[1], z / ff_shape[2]])

                orthant_ind = np.array([
                    x * num_segments // ff_shape[0],
                    y * num_segments // ff_shape[1],
                    z * num_segments // ff_shape[2]
                ])

                # find containing tetrahedron
                factors = None
                vertex_offsets = None
                for i, tetrahedron in enumerate(tetrahedron_offsets):
                    p, q, r, s = [(vertex + orthant_ind) / num_segments for vertex in tetrahedron]
                    if tetrahedron_contains(p, q, r, s, coord):
                        factors = barycentric_3d(p, q, r, s, coord)
                        vertex_offsets = tetrahedron
                        break
                else:
                    print("Could not generate ff blend factors", orthant_ind, coord)
                    sys.exit(1)

                blend = torch.zeros(key_points).to(device)
                for factor, vertex in zip(factors, vertex_offsets):
                    overall_vertex = vertex + orthant_ind
                    overall_index = overall_vertex[2] + overall_vertex[1] * (num_segments + 1) + overall_vertex[2] * (num_segments + 1) ** 2
                    blend[overall_index] = factor

                ret[x][y][z] = blend
    return ret
"""

"""
# barycentric interpolation in 3d
class R3BarycentricFFTransformer(FFTransformer):
    def __init__(self, ff_shape, subdivisions):
        ff_shape: shape of the feature field's underlying data (x, y, z)
        subdivisions: the more subdivisions, the higher frequency the local transformation
                      beyond a certain frequency, the transformation may not even 
                      considered smooth 

        super().__init__(r3_blending_matrix(ff_shape, subdivisions));
"""
