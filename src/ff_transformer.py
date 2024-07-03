import torch
import torch.nn as nn
import sys
from utils import *
from config import ONLY_IDENTITY_COMPONENT

device = get_device()


# Feature Field Transformer
# Given the group action elements, actually apply a local transformation to
# the target manifold feature field
class FFTransformer:
    def __init__(self, blend_factors):
        """
            blend_factors: tensor that is of shape [*manifold_shape, num_sample_points] denoting the contribution of each key point to each manifold point
        """
        self.blend_factors = blend_factors

    def num_key_points(self):
        return self.blend_factors.shape[-1]

    def apply(self, cosets, group_key_points, feature_field):
        """
            cosets: None or tensor of shape [(batch), ff_dimension, ff_dimension]
            group_key_points: None or elements of the lie algebra of shape [(batch), num_key_points, ff_dim, ff_dim]
            feature_field: the actual input value tensor of shape [(batch), *manifold_shape, *ff_shape]
        """
        assert cosets is not None or group_key_points is not None

        # mimic num_key_points
        if cosets is not None:
            cosets = cosets.unsqueeze(-3)

        # add manifold size dimension
        manifold_size = self.blend_factors.shape[:-1]
        for i in range(len(manifold_size)):
            # before two matrix dimensions and the num_key_points dimension
            if group_key_points is not None:
                group_key_points = group_key_points.unsqueeze(-4)
            if cosets is not None:
                cosets = cosets.unsqueeze(-3)

        if group_key_points is not None:
            # shape: [(batch), *manifold_size, num_key_points, ff_dimension, ff_dimension]
            mult = group_key_points * self.blend_factors.unsqueeze(-1).unsqueeze(-1)
            # shape: [(batch), *manifold_size, ff_dimension, ff_dimension]
            matrices = torch.matrix_exp(torch.sum(mult, dim=-3))
        else:
            matrices = None

        if ONLY_IDENTITY_COMPONENT:
            return (matrices @ feature_field.unsqueeze(-1)).squeeze(-1)
        else:
            if matrices is None:
                build = cosets
            elif cosets is None:
                build = matrices
            else:
                build = cosets @ matrices

            return (build @ feature_field.unsqueeze(-1)).squeeze(-1)

    # ff transformer can also just be used to create smooth functions in general
    # this method is useful in the case of generating smooth vector field for winding number problem
    # value at kp: [bs (exactly one dimension), num_key_points, *shape]
    # ret [bs, *manifold_shape, *shape]
    def smooth_function(self, value_at_key_points):
        blend_factors = self.blend_factors.unsqueeze(0)
        for i in range(len(value_at_key_points.shape) - 2):
            blend_factors = blend_factors.unsqueeze(-1)

        manifold_size = self.blend_factors.shape[:-1]
        for i in range(len(manifold_size)):
            value_at_key_points = value_at_key_points.unsqueeze(1)

        # shape: [bs, *manifold_size, num_key_points, *shape]
        mult = value_at_key_points * blend_factors
        # shape: [bs, *manifold_size, *shape]
        mult = torch.sum(mult, dim=len(manifold_size) + 1)
        return mult

class TorusFFTransformer(FFTransformer):
    def __init__(self, u_dim, v_dim, u_keypoints, v_keypoints):
        assert u_dim % u_keypoints == 0
        assert v_dim % v_keypoints == 0

        num_kp = u_keypoints * v_keypoints
        blend = torch.zeros((u_dim, v_dim, num_kp)).to(device)

        for u in range(u_dim):
            for v in range(v_dim):
                # bi linear interpolate off of four closest points
                u0 = u * u_keypoints // u_dim
                u1 = (u0 + 1) % u_keypoints
                v0 = v * v_keypoints // v_dim
                v1 = (v0 + 1) % v_keypoints

                s = (u - (u0 * u_dim // u_keypoints)) / (u_dim // u_keypoints)
                t = (v - (v0 * v_dim // v_keypoints)) / (v_dim // v_keypoints)

                blend[u, v, u0 * v_keypoints + v0] = (1 - s) * (1 - t)
                blend[u, v, u1 * v_keypoints + v0] = s * (1 - t)
                blend[u, v, u0 * v_keypoints + v1] = (1 - s) * t
                blend[u, v, u1 * v_keypoints + v1] = s * t

        super().__init__(blend)


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


# barycentric interpolation in 3d
class R3BarycentricFFTransformer(FFTransformer):
    def __init__(self, ff_shape, subdivisions):
        """
            ff_shape: shape of the feature field's underlying data (x, y, z)
            subdivisions: the more subdivisions, the higher frequency the local transformation
                beyond a certain frequency, the transformation may not even 
                considered smooth 
        """

        super().__init__(r3_blending_matrix(ff_shape, subdivisions));
