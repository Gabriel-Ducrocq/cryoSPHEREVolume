import torch
import numpy as np
from torch import nn

def cylinder(x,y,z):
    if x**2 + y**2 < 1 and z > 0 and z < 1:
        return 1

    return 0

x = y = z = np.linspace(0, 1, 100)
xx, yy, zz = np.meshgrid(x, y, z)

coordinates = np.stack((xx, yy, zz), axis=-1)
rot_mat = np.eye(3)

rotated_coordinates = np.einsum("ij, bklj -> bkli", rot_mat, coordinates)


def project(axis, n_points, func):
    pass

class Projector(nn.Module):
    def __init__(self, n_points = 128):
        self.n_points = n_points

    def project(self,):
        pass