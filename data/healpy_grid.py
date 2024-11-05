import json
import argparse
import healpy
import numpy as np


def compute_healpix_grid(Nside):
    """
    Compute the angles associated with the pixels of the Healpix grid, in nested format.
    :param Nside: integer, nside of the grid, see Healpix.
    """
    assert np.log2(nside) == np.floor(np.log2(nside)), "Nside must be a power of 2."
    Npix = 12 * Nside * Nside
    print(f"Generating grid with {Npix} pixels")
    theta, phi = healpy.pix2ang(Nside, np.arange(Npix), nest=True, lonlat=False)
    return theta.tolist(), phi.tolist()

def save_grid(quaternions, Nside, output_path):
    np.save(output_path + f"{Nside}_nside.npy", quaternions)


def hopf_to_quat(theta, phi, psi):
    """
    Hopf coordinates to quaternions
    theta: [0,pi)
    phi: [0, 2pi)
    psi: [0, 2pi)
    """
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    quat = np.array(
        [
            ct * np.cos(psi / 2),
            ct * np.sin(psi / 2),
            st * np.cos(phi + psi / 2),
            st * np.sin(phi + psi / 2),
        ]
    )
    return quat.T.astype(np.float32)


def grid_s2(resol):
    Nside = 2**resol
    Npix = 12 * Nside * Nside
    theta, phi = healpy.pix2ang(Nside, np.arange(Npix), nest=True)
    return theta, phi

def grid_SO3(resol):
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(
        np.repeat(theta, len(psi)),  # repeats each element by len(psi)
        np.repeat(phi, len(psi)),  # repeats each element by len(psi)
        np.tile(psi, len(theta)),
    )  # tiles the array len(theta) times
    return quat  # hmm convert to rot matrix?


def grid_s1(resol):
    Npix = 6 * 2**resol
    dt = 2 * np.pi / Npix
    grid = np.arange(Npix) * dt + dt / 2
    return grid


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--nside', type=int, required=True)
    parser_arg.add_argument('--output_path', type=str, required=True)
    args = parser_arg.parse_args()
    nside = args.nside
    output_path = args.output_path

    quaternions = grid_SO3(nside)
    save_grid(quaternions, nside, output_path)
