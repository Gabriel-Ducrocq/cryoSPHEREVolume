import json
import argparse
import healpy
import numpy as np
import torch


with open(f"data/healpy_grid.json") as hf:
    _GRIDS = {int(k): np.array(v).T for k, v in json.load(hf).items()}


def get_s2_neighbor(mini, cur_res):
    """
    Return the 4 nearest neighbors on S2 at the next resolution level.
    :param mini: np.array indices of the pixels on s2
    :current res: integer, current resolution
    return np.array of the coordinates of the neighboring pixels, np array, their indices.
    """
    Nside = 2 ** (cur_res + 1)
    ind = np.arange(4) + 4 * mini
    print("NSIDE", Nside)
    print("IND", ind)
    return healpy.pix2ang(Nside, ind, nest=True), ind


def get_s2_neighbor_tensor(mini, cur_res):
    """
    Return the 4 nearest neighbors on S2 at the next resolution level

    mini: [nq] np array of the current location on SO(2)

    output: [2, nq, 4] (np.array) containing for each original point the angles corresponding to its children in angles as first dimensions, [nq, 4] (np.array) of new indices
    """
    n_side = 2 ** (cur_res + 1)
    #Since we use a nested scheme for healpix, each grid point is divided into 4 points.
    ind = np.arange(4) + 4 * mini[..., None]
    return pix2ang_tensor(n_side, ind, nest=True), ind


def pix2ang_tensor(n_side, i_pix, nest=False, lonlat=False):
    """
    i_pix: [nq, 4] the array of new location indices to query

    output: [2, nq, 4] (np.array)
    """
    assert n_side in _GRIDS, f"n_side {n_side} is not on the precomputed grid resolutions"
    assert _GRIDS is not None and nest and not lonlat
    # _GRIDS[n_side]: [x, 2]
    #We get the number of locations on S2 to query
    nq = i_pix.shape[0]
    #We first flatten i_pix to get all the indexes in one dimension, we then query the right grid with correct (new) resolution
    #Then we reshape (nq, 4, 2) to get, for each of the original locations on s2, the angles associated with each one of its child location.
    #Finally we transpose the tensor to get the two angles as first dimension (2, nq, 4)
    return np.einsum('ijk->kij', _GRIDS[n_side][i_pix.reshape(-1)].reshape(nq, 4, 2))


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


def get_s1_neighbor_tensor(mini, curr_res):
    """
    Return the 2 nearest neighbors on S1 at the next resolution level

    mini: [nq] the current indices on s1

    output: [nq, 4] (np.array) of locations on s1, [nq, 4] (np.array) of indices of the new locations on s1
    """
    n_pix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / n_pix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping
    ind = np.repeat(2 * mini[..., None] - 1, 4, axis=-1) + np.arange(4)
    ind[ind < 0] += n_pix
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


def get_neighbor_tensor(quat, q_ind, cur_res, device):
    """
    quat: [nq, 4] of quaternions at the current locations.
    q_ind: [nq, 2], np.array of indices of the locations on SO(3)
    cur_res: int

    output: [nq, 8, 4] np.array of 8 new rotation for each of nq sample x poses we kept, [nq, 8, 2] (np.array) same for the indices.
    """
    nq = quat.shape[0]

    (theta, phi), s2_next = get_s2_neighbor_tensor(q_ind[..., 0], cur_res)
    psi, s1_next = get_s1_neighbor_tensor(q_ind[..., 1], cur_res)
    #Compute the quaternions from the Hopf fibration coordinates
    quat_n = hopf_to_quat_tensor(
        np.repeat(theta[..., None], psi.shape[-1], axis=-1).reshape(nq, -1),
        np.repeat(phi[..., None], psi.shape[-1], axis=-1).reshape(nq, -1),
        np.repeat(psi[:, None], theta.shape[-1], axis=-2).reshape(nq, -1)
    )  # nq, 16, 4
    #Concatenate the indexes on s2 and s1 to get the position on the Hopf fibration.
    ind = np.concatenate([
        np.repeat(s2_next[..., None], psi.shape[-1], axis=-1).reshape(nq, -1)[..., None],
        np.repeat(s1_next[:, None], theta.shape[-1], axis=-2).reshape(nq, -1)[..., None]
    ], -1)  # nq, 16, 2

    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    print("QUAT N", quat_n.shape)
    print("QUAT", quat.shape)
    quat_n = torch.tensor(quat_n).to(device)
    dists = torch.minimum(
        torch.sum((quat_n - quat[:, None]) ** 2, dim=-1),
        torch.sum((quat_n + quat[:, None]) ** 2, dim=-1)
    )  # nq, 16
    ii = torch.argsort(dists, dim=-1)[:, :8].cpu()
    quat_out = quat_n[torch.arange(nq)[..., None], ii]
    ind_out = ind[torch.arange(nq)[..., None], ii]
    return quat_out, ind_out


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

def save_grid(quaternions, ind, resol, output_path):
    np.save(output_path + f"{resol}_resol_quat.npy", quaternions)
    np.save(output_path + f"{resol}_resol_ind.npy", ind)
    #d = {"quat":quaternions, "indexes":ind, "resol":resol}
    #with open(output_path + f"{resol}_resol.json", "w") as f:
    #    json.dump(d, f)


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


def hopf_to_quat_tensor(theta, phi, psi):
    """
    Hopf coordinates to quaternions
    theta: [nq, 16] (np.array), [0,pi)
    phi: [nq, 16] (np.array), [0,2pi)
    psi: [nq, 16] (np.array), [0,2pi)

    output: [nq, 16, 4]
    """
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    quat = np.concatenate([
        (ct * np.cos(psi / 2))[..., None],
        (ct * np.sin(psi / 2))[..., None],
        (st * np.cos(phi + psi / 2))[..., None],
        (st * np.sin(phi + psi / 2))[..., None]
    ], -1)
    return quat.astype(np.float32)


def grid_s2(resol):
    Nside = 2**resol
    Npix = 12 * Nside * Nside
    theta, phi = healpy.pix2ang(Nside, np.arange(Npix), nest=True)
    return theta, phi

def grid_SO3(resol):
    theta, phi = grid_s2(resol)
    ind_s2 = np.arange(len(theta))
    psi = grid_s1(resol)
    ind_s1 = np.arange(len(psi))
    quat = hopf_to_quat(
        np.repeat(theta, len(psi)),  # repeats each element by len(psi)
        np.repeat(phi, len(psi)),  # repeats each element by len(psi)
        np.tile(psi, len(theta)),
    )  # tiles the array len(theta) times

    ind = np.concatenate([np.repeat(ind_s2, len(psi))[:, None], np.tile(ind_s1, len(theta))[:, None]], axis=-1)
    return quat, ind  # hmm convert to rot matrix?


def grid_s1(resol):
    Npix = 6 * 2**resol
    dt = 2 * np.pi / Npix
    grid = np.arange(Npix) * dt + dt / 2
    return grid


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--resol', type=int, required=True)
    parser_arg.add_argument('--output_path', type=str, required=True)
    args = parser_arg.parse_args()
    resol = args.resol
    output_path = args.output_path

    quaternions, ind = grid_SO3(resol)
    save_grid(quaternions, ind, resol, output_path)
