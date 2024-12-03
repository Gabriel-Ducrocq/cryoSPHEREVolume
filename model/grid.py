import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


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
        #Since the theoretical Fourier transform is periodic of period 1/apix, it means we are sampling on an interval [-1/2apix, 1/2apix] for odd size or
        # [-1/2apix, -1/2apix[ for even sizes. In any case, the highest frequency in absolue value is 1/2apix. So we should divide by 1/2apix to recover an interval [-1, 1].
        #Since we apply a fftshift, the 0 frequency is at the center.
        if side_shape % 2 == 0:
            extent = 1/(2*apix)
            ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))/extent
        else:
            extent = (side_shape-1)/(2*side_shape*apix)
            ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))/extent

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

        if side_shape % 2 == 0:
            extent = 1/(2*apix)
            ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))/extent
        else:
            extent = (side_shape-1)/(2*side_shape*apix)
            ax = torch.fft.fftshift(torch.fft.fftfreq(self.side_shape, self.apix))/extent

        self.extent = np.abs(ax[0])
        mx, my = torch.meshgrid(ax, ax, indexing="xy")
        self.freqs = torch.stack([mx.flatten(), my.flatten(), torch.zeros(side_shape**2, dtype=torch.float32)], 1)
        self.freq_radiuses = torch.sqrt(torch.sum(self.freqs[:, :2]**2, axis=-1))
        mask = self.freq_radiuses < self.radius/(self.side_shape//2) * self.extent
        self.register_buffer("mask", mask.to(device))

        mx, my, mz = torch.meshgrid(ax, ax, ax, indexing="xy")
        freqs_volume = torch.stack([mx.flatten(), my.flatten(), mz.flatten()], 1)
        radiuses_volume = torch.sqrt(torch.sum(freqs_volume[:, :2]**2, axis=-1))
        mask_volume = radiuses_volume < self.radius/(self.side_shape//2) * self.extent
        self.register_buffer("mask_volume", mask_volume.to(device))

        self.masks_2d = {}
        self.masks_2d[self.radius] = mask

    def get_mask(self, radius):
        if radius in self.masks_2d:
            return self.masks_2d[radius]

        assert 2*radius + 1 < self.side_shape, f"Mask with radius {radius} is too large for image of size {self.side_shape}x{self.side_shape}."
        mask = self.freq_radiuses < radius/(self.side_shape//2) * self.extent
        self.masks_2d[radius] = mask
        return mask

    def plot_mask(self, radius):
        if radius in self.masks_2d:
            mask = self.masks_2d[radius]
        else:
            mask = self.get_mask(radius)

        mask = mask.reshape(self.side_shape, self.side_shape)
        plt.imshow(mask)
        plt.show()




class PositionalEncoding(nn.Module):

    def __init__(self, pe_dim=6, D=128, pe_type="geo", include_input=False):
        """
        Initilization of a positional encoder.

        Parameters
        ----------
        num_encoding_functions: int
        include_input: bool
        normalize: bool
        input_dim: int
        gaussian_pe: bool
        gaussian_std: float
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.include_input = include_input
        self.pe_type = pe_type
        self.D = D

        if self.pe_type == "gau1":
            self.gaussian_weights = nn.Parameter(torch.randn(3 * self.pe_dim, 3) * D / 4, requires_grad=False)
        elif self.pe_type == "gau2":
            # FTPositionalDecoder (https://github.com/zhonge/cryodrgn/blob/master/cryodrgn/models.py)
            # __init__():
            #   rand_freqs = randn * 0.5
            # random_fourier_encoding():
            #   freqs = rand_freqs * coords * D/2
            # decode()/eval_volume():
            #   extent < 0.5 -> coords are in (-0.5, 0.5), while in cryostar coords are in (-1, 1)
            self.gaussian_weights = nn.Parameter(torch.randn(3 * self.pe_dim, 3) * D / 8, requires_grad=False)
        elif self.pe_type == "geo1":
            # frequency: (1, D), wavelength: (2pi/D, 2pi)
            f = D
            self.frequency_bands = nn.Parameter(f * (1. / f)**(torch.arange(self.pe_dim) / (self.pe_dim - 1)),
                                                requires_grad=False)
        elif self.pe_type == "geo2":
            # frequency: (1, D*pi)
            f = D * np.pi
            self.frequency_bands = nn.Parameter(f * (1. / f)**(torch.arange(self.pe_dim) / (self.pe_dim - 1)),
                                                requires_grad=False)
        elif self.pe_type == "no":
            pass
        else:
            raise NotImplemented

    def __repr__(self):
        return str(self.__class__.__name__) + f"({self.pe_type}, num={self.pe_dim})"

    def out_dim(self):
        if self.pe_type == "no":
            return 3
        else:
            ret = 3 * 2 * self.pe_dim
            if self.include_input:
                ret += 3
            return ret

    def forward(self, tensor):
        """
        tensor is a torch tensor of size (batch_size, n_coordinates, 3)
        return a tensor of shape (batch_size, n_coordinates, 3*pe_dimension) if self.include input is False, else (batch_size, n_coordinates, 3*pe_dimension + 3)
        """
        with torch.autocast("cuda", enabled=False):
            assert tensor.dtype == torch.float32
            if self.pe_type == "no":
                return tensor

            encoding = [tensor] if self.include_input else []
            if "gau" in self.pe_type:
                x = torch.matmul(tensor, self.gaussian_weights.T)
                encoding.append(torch.cos(x))
                encoding.append(torch.sin(x))
            elif "geo" in self.pe_type:
                bsz, num_coords, _ = tensor.shape
                x = self.frequency_bands[None, None, None, :] * tensor[:, :, :, None]
                x = x.reshape(bsz, num_coords, -1)
                encoding.append(torch.cos(x))
                encoding.append(torch.sin(x))

            ret = torch.cat(encoding, dim=-1)

        return ret











