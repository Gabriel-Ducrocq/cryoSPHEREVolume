import json
import argparse
import healpy
import numpy as np



def get_s2_neighbor(mini, cur_res):
    """
    Return the 4 nearest neighbors on S2 at the next resolution level.
    :param mini: np.array indices of the pixels on s2
    :current res: integer, current resolution
    return np.array of the coordinates of the neighboring pixels, np array, their indices.
    """
    Nside = 2 ** (cur_res + 1)
    ind = np.arange(4) + 4 * mini
    return pix2ang(Nside, ind, nest=True), ind


def get_s1_neighbor(mini, curr_res):
    """
    Return the 2 nearest neighbors on S1 at the next resolution level.
    :param mini: np.array, indices on s1 of the current points.
    :param curr_res: int, current resolution.
    """
    Npix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / Npix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping
    ind = np.arange(2 * mini - 1, 2 * mini + 3)
    if ind[0] < 0:
        ind[0] += Npix
    return ind * dt + dt / 2, ind


def get_neighbor(quat, s2i, s1i, cur_res):
    """
    Return the 8 nearest neighbors on SO3 at the next resolution level
    :param quat: torch.tensor(N_points, 4) quaternions corresponding to the current poses grid.
    :param s2i: np.array(N_points,) indexes of the current poses on the s2 sphere
    :param s1i: np.array(N_point, ) indexes of the current poses on the s1 sphere
    :param cur_res: integer, current resolution.
    return np.array(N_points*8, 4) of quaternions corresponding to the nearest neighbors, np.array(N_points*8,2) of the indices on s2 and s1 of the nearest neighbors.
    """
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, cur_res)
    psi, s1_nexti = get_s1_neighbor(s1i, cur_res)
    #Get the new quaternions
    quat_n = hopf_to_quat(
        np.repeat(theta, len(psi)), np.repeat(phi, len(psi)), np.tile(psi, len(theta))
    )
    #Get the new set of indices
    ind = np.array([np.repeat(s2_nexti, len(psi)), np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q, since +q and -q correspond to the same rotation
    dists = np.minimum(
        np.sum((quat_n - quat) ** 2, axis=1), np.sum((quat_n + quat) ** 2, axis=1)
    )
    #Take the 8 smallest neighbors.
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]


def get_so3_neighbours(pixel_indices, current_resolution, n_neighb=8):
    """
    This function takes a set of pixels and gives back the indices of the closest n_neighb in the next subdivision of the so3 grid.
    """


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
