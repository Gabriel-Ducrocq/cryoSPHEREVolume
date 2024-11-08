from tqdm import tqdm
from time import time
import model.utils
from model import utils
from model import loss
from model import renderer
from model.grid import Mask
import torch
import model
import numpy as np
from time import time
from model.grid import Grid, rotate_grid
from decorators import timing
from pytorch3d.transforms import quaternion_to_matrix



@timing
def precompute_wigner_D(wigner_calculator, poses, l_max, device="cpu"):
	"""
	This function pre compute the wigner D matrices for all the given poses, since it seems to be the most expensive step in the loop.
	:param poses: torch.tensor(batch_size, 3, 3) rotation matrices corresponding to the poses.
	"""
	all_wigner = {}
	for i, pose in enumerate(poses):
		wigner_pose = wigner_calculator.compute_wigner_D(l_max, pose[None, :, :].to(device), device)
		all_wigner[i] = wigner_pose

	return all_wigner



class PoseSearch:
	def __init__(self, kmin, kmax, all_wigner, base_resol=1, total_iter=5, base_resol=1):
		"""
		Performs a pose search
		:param kmin: integer, minimum resolution in frequency marching, expressed in number of pixels.
		:param kmax: integer, maximum resolution in frequency marching, expressed in number of pixels.
		:param base_resol: integer, minimal resolution at which we start the grid seach on SO(3)
		:param all_wigner: nested dictionnary, {base_resol:{pixel1_index: ...}} of precomputed wigner D matrices for each pixel for the base resol. We later update it to cache
							the computed wigner D matrices at higher resolutions.
		"""
		self.kmin = kmin
		self.kmax = kmax
		self.resol = resol
		self.total_iter
		self.resolution_run
		self.mask = Mask(Npix_downsize, apix_downsize)
		self._so3_neighbor_cache = {}
		self.all_wigner = all_wigner

	def get_frequency_limit(n_iter):
		return min(kmin + int((n_iter/self.total_iter) * (self.kmax - kmin)), sef.resolution_run)

	def apply_mask(images, k):
		"""
		Applies mask to the images
		:param images: torch.tensor(batch_size, side_shape**2)
		:param k: float, resolution cutoff in pixels.
		"""
		mask = self.mask.get_mask(k)
		return images[:, mask]


    def get_neighbor_so3(self, quat, s2i, s1i, res):
        """
        This function gets the 8 nearest neighbors on so(3) and caches the results for later use.
        :param quat: torch.tensor(N_points, 4) of quaternions at the current points
        :param s2i: np.array(N_points,) of indices of the current points on s2
        :param s1i: np.array(N_points,) of indices of the current points on s1
        :param res: integer, current resolution.
        """
        key = (int(s2i), int(s1i), int(res))
        if key not in self._so3_neighbor_cache:
            self._so3_neighbor_cache[key] = so3_grid.get_neighbor(quat, s2i, s1i, res)
        # FIXME: will this cache get too big? maybe don't do it when res is too
        return self._so3_neighbor_cache[key]

	def subdivide(self, quat: np.ndarray, q_ind: np.ndarray, cur_res: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Subdivides poses for next resolution level

        Inputs:
            quat (N x 4 tensor): quaternions
            q_ind (N x 2 np.array): index of current S2xS1 grid
            cur_res (int): Current resolution level

        Returns:
            quat  (N x 8 x 4) np.array of quaternions at the new (neighbors) points.
            q_ind (N x 8 x 2) np.array of indices of the new (neighbors) points.
            rot   (N*8 x 3 x 3) tensor of rotation matrices corresponding to each of quaternions in quat.
        """
        N = quat.shape[0]

        assert len(quat.shape) == 2 and quat.shape == (N, 4), quat.shape
        assert len(q_ind.shape) == 2 and q_ind.shape == (N, 2), q_ind.shape

        # get neighboring SO3 elements at next resolution level -- todo: make this an array operation
        neighbors = [
            self.get_neighbor_so3(quat[i], q_ind[i][0], q_ind[i][1], cur_res)
            for i in range(len(quat))
        ]
        quat = np.array([x[0] for x in neighbors])  # Bx8x4
        q_ind = np.array([x[1] for x in neighbors])  # Bx8x2

        rot = quaternion_to_matrix(torch.from_numpy(quat).view(-1, 4)).to(
            self.device
        )

        assert len(quat.shape) == 3 and quat.shape == (N, 8, 4), quat.shape
        assert len(q_ind.shape) == 3 and q_ind.shape == (N, 8, 2), q_ind.shape
        assert len(rot.shape) == 3 and rot.shape == (N * 8, 3, 3), rot.shape

        return quat, q_ind, rot

    def search(alms_per_coordinate, true_images, spherical_harmonics, l_max, device, radius_indexes, ctf):
    	"""
    	Perform the pose search with frequency marching
    	:param alms_per_coordinate: torch.tensor(batch_size, N_pixels_in_mask, (l_max+1)**2)
		"""
		batch_size = true_images.shape[0]
		for n_it in range(1, self.total_iter+1):
			resol = self.base_resol + (n_it -1) #n_it start at 1 but we want our resol to start at 1
			k = self.get_frequency_limit(n_it)
			assert resol in self.all_wigner, "resolution"
			all_wigners = self.all_wigner[resol]
			mask = self.mask.get_mask(k)
			rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
			predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate, rotated_spherical_harmonics, mask, radius_indexes, device)
			if use_ctf:
		    	batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
			else:
		    	batch_predicted_images = predicted_images








def perform_pose_search(batch_translated_images_hartley, grid_wigner, latent_mean, latent_std, spherical_harmonics, experiment_settings, tracking_metrics, alms_per_coordinate, 
	circular_mask, radius_indexes, ctf, use_ctf, poses, l_max, device, wigner_calculator, indexes):
	batch_size = latent_mean.shape[0]
	npix = batch_translated_images_hartley.shape[-1]
	reconstruction_errors = torch.ones(batch_size, dtype=torch.float32, device=device)*torch.inf
	poses_min = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
	poses_min[:, 0, 0] = poses_min[:, 1, 1] = poses_min[:, 2, 2] = 1
	argmin_images = torch.zeros(batch_size, int(np.sqrt(npix)), int(np.sqrt(npix)),  dtype=torch.float32, device=device)
	for k, batch_poses in enumerate(tqdm(poses)):
		all_wigner = grid_wigner[k]
		start = time()
		batch_poses = batch_poses[None, :, :].repeat(batch_size, 1, 1).to(device)
		start_apply = time()
		rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
		end_apply = time()
		print("Applying Wigner D:", end_apply - start_apply)
		start_predict_image = time()
		predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate, rotated_spherical_harmonics, circular_mask.mask, radius_indexes, device)
		end_predicted_images = time()
		print("Predicting images:", end_predicted_images - start_predict_image)
		if use_ctf:
		    batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
		else:
		    batch_predicted_images = predicted_images

		#Instead of averaging over the batch dimension, we keep it to know which pose minimizes the reconstruction error for each pose.
		losses = torch.mean((batch_translated_images_hartley - batch_predicted_images.flatten(start_dim=1, end_dim=2))**2, dim=-1)
		poses_min[losses < reconstruction_errors] = batch_poses[losses < reconstruction_errors]
		reconstruction_errors[losses < reconstruction_errors] = losses[losses < reconstruction_errors]
		argmin_images[losses < reconstruction_errors] = batch_predicted_images[losses < reconstruction_errors]
		end = time()
		print("Total time:", end-start)

	return poses_min, reconstruction_errors, argmin_images