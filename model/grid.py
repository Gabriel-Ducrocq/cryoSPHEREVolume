import torch
import numpy as np


def rotate_grid(rotation_matrices, grid):
    """
    Compute the frequencies on the rotated slice to apply the Fourier slice theorem
    :param rotation_matrices: torch.tensor(N_batch, 3, 3) of rotation matrix in RIGHT multiplication convention
    :param grid: torch.tensor(side_shape**2, 3) of frequencies.
    :return: torch.tensor(N_batch, side_shape**2, 3) of rotated frequencies
    """
    return torch.einsum("b k q, s q -> b s k", rotation_matrices, grid)
class Grid(torch.nn.Module):
    """
    Class describing the ctf, built from starfile
    """
    def __init__(self, side_shape, apix, device="cpu"):
        super().__init__()
        self.apix = apix
        self.side_shape = side_shape
        ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))
        mx, my = torch.meshgrid(ax, ax, indexing="xy")
        freqs = torch.stack([mx.flatten(), my.flatten(), torch.zeros(side_shape**2, dtype=torch.float32)], 1)
        self.register_buffer("freqs", freqs)
        self.freqs = self.freqs.to(device)


        mx, my, mz = torch.meshgrid(ax, ax, ax, indexing="xy")
        freqs = torch.stack([mx.flatten(), my.flatten(), mz.flatten()], 1)
        self.register_buffer("freqs_volume", freqs)
        self.freqs_volume = self.freqs_volume.to(device)


class Mask(torch.nn.Module):
    """
    Class describing the circular mask in Fourier space
    """
    def __init__(self, side_shape, apix, radius=None, device="cpu"):
        """
        :param side_shape: integer, number of pixels a side.
        :param apix: float, size of a pixel.
        :param radius: float, radius of the mask, in pixels.
        """
        super().__init__()
        if radius is None:
            radius = side_shape//2

        self.radius = radius
        self.side_shape = side_shape
        self.apix = apix
        ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))
        extent = np.abs(ax[0])
        mx, my = torch.meshgrid(ax, ax, indexing="xy")
        freqs = torch.stack([mx.flatten(), my.flatten(), torch.zeros(side_shape**2, dtype=torch.float32)], 1)
        radiuses = torch.sum(freqs[:, :2]**2, axis=-1)
        mask = radiuses < self.radius/self.side_shape * extent
        self.register_buffer("mask", mask.to(device))





