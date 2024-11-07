from tqdm import tqdm
from time import time
import model.utils
from model import utils
from model import loss
from model import renderer
import torch
import model
import numpy as np
from time import time
from model.grid import Grid, rotate_grid
from decorators import timing



@timing
def precompute_wigner_D(wigner, calculator, poses, l_max, device="cpu"):
	"""
	This function pre compute the wigner D matrices for all the given poses, since it seems to be the most expensive step in the loop.
	:param poses: torch.tensor(batch_size, 3, 3) rotation matrices corresponding to the poses.
	"""
	all_wigner = {}
	for pose in poses:
		wigner_pose = wigner_calculator.compute_wigner_D(l_max, pose, device)
		all_wigner[pose] = all_wigner_pose

	return all_wigner




def perform_pose_search(batch_translated_images_hartley, grid_wigner, latent_mean, latent_std, spherical_harmonics, experiment_settings, tracking_metrics, alms_per_coordinate, 
	circular_mask, radius_indexes, ctf, use_ctf, poses, l_max, device, wigner_calculator, indexes):
	batch_size = latent_mean.shape[0]
	npix = batch_translated_images_hartley.shape[-1]
	reconstruction_errors = torch.ones(batch_size, dtype=torch.float32, device=device)*torch.inf
	poses_min = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
	poses_min[:, 0, 0] = poses_min[:, 1, 1] = poses_min[:, 2, 2] = 1
	argmin_images = torch.zeros(batch_size, int(np.sqrt(npix)), int(np.sqrt(npix)),  dtype=torch.float32, device=device)
	for batch_poses in tqdm(poses):
		all_wigner = grid_wigner[batch_poses]
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