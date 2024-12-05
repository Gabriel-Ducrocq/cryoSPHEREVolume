import torch
import einops
import numpy as np
from time import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

def primal_to_fourier2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier2d_to_primal(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r

def apply_ctf(fourier_images, ctf, indexes):
    """
    apply ctf to images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    ctf: CTF object
    indexes: torch.tensor(batch_size, type=int)
    return ctf corrupted images
    """
    print("CTF VALUES:", ctf.compute_ctf(indexes))
    fourier_images *= -ctf.compute_ctf(indexes)
    return fourier_images


class SpatialGridTranslate(torch.nn.Module):
    """
    Class that defines the way we translate the images in real space.
    """
    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        #Coord is of shape (N_coord, 2). The coordinates go from -1 to 1, representing not actual physical coordinates but rather proportion of a half image.
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Supposing that D is 96, a point is at 0.0:
                - adding 48 should move it to the right corner which is 1.0
                    1.0 = 0.0 + 48 / (96 / 2)
                - adding 96(>48) should leave it at 0.0
                    0.0 = 0.0 + 96 / (96 / 2) - 2.0
                - adding -96(<48) should leave it at 0.0
                    0.0 = 0.0 - 96 / (96 / 2) + 2.0

            Input:
                images: (B, N_pix, N_pix)
                trans:  (B, T,  2)

            Returns:
                images: (B, N_pix, N_pix)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]
        #We translate the coordinates not in terms of absolute translations but in terms of fractions of a half image, to be consistent with the way coord is defined.
        grid = einops.rearrange(self.coords, "N C2 -> 1 1 N C2") - \
            einops.rearrange(trans, "B T C2 -> B T 1 C2") * 2 / self.D
        grid = grid.flip(-1)  # convert the first axis from slow-axis to fast-axis
        grid[grid >= 1] -= 2
        grid[grid <= -1] += 2
        grid.clamp_(-1.0, 1.0)

        #We sample values at coordinates given by grid, using bilinear interpolation with padding mode zero.
        sampled = F.grid_sample(einops.rearrange(images, "B NY NX -> B 1 NY NX"), grid, align_corners=True)

        sampled = einops.rearrange(sampled, "B 1 T (NY NX) -> B T NY NX", NX=NX, NY=NY)
        return sampled[:, 0, :, :]




        



 