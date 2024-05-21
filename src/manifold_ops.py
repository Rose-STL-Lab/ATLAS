import torch.nn as nn
import sys
from abc import ABC, abstractmethod
from utils import *

device = get_device()

# Feature Field Transformer
# Given the group action elements, actually apply a local transformation to
# the target manifold feature fiedl
class FFTransformer(ABC):
    def __init__(self, blend_factors):
        """
            blend_factors: tensor that is of shape [manifold_size, num_sample_points] denoting the contribution of each basis element at each manifold point
        """
        self.blend_factors = blend_factors

    def num_key_points(self):
        return self.blend_factors.shape[-1]

    # cosets should be regular elements
    # group_key_points should be in the tangent space of the identity
    def apply(self, cosets, group_key_points, feature_field):
        # add manifold size dimension
        manifold_size = self.blend_factors.shape[:-1]
        for i in range(len(manifold_size)):
            # before two matrix dimensions and the num_key_points dimension
            group_key_points = group_key_points.unsqueeze(-4)
            cosets = cosets.unsqueeze(-3)

        # shape: [(batch), *manifold_size, num_key_points, ff_dimension, ff_dimension]
        mult = group_key_points * self.blend_factors.unsqueeze(-1).unsqueeze(-1)
        # shape: [(batch), *manifold_size, ff_dimension, ff_dimension]
        matrices = torch.matrix_exp(torch.sum(mult, dim=-3))

        return (cosets @ matrices @ feature_field.unsqueeze(-1)).squeeze(-1)

# applies via barycentric interpolation
class SphereUVFFTransformer(FFTransformer):
    pass 


def r3_blending_matrix(ff_shape, subdivisions):
    num_segments = 2 ** subdivisions
    key_points = (num_segments + 1) ** 3
    ret = torch.empty((*ff_shape, key_points))

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

# barycentric interpolation in 4d
class R3FFTransformer(FFTransformer):
    def __init__(self, ff_shape, subdivisions):
        """
            ff_shape: shape of the feature field's underlying data (x, y, z)
            subdivisions: the more subdivisions, the higher frequency the local transformation
                beyond a certain frequency, the transformation may not even 
                considered smooth 
        """

        super().__init__(r3_blending_matrix(ff_shape, subdivisions));
