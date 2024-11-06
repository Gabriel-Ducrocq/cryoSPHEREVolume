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




def perform_pose_search(batch_translated_images_hartley, latent_mean, latent_std, spherical_harmonics, experiment_settings, tracking_metrics, alms_per_coordinates, 
	circular_mask, radius_indexes, ctf, use_ctf, poses, lmax, device):
	batch_size = latent_mean.shape[0]
	npix = batch_translated_images_hartley.shape[-1]
	reconstruction_errors = torch.ones(batch_size, dtype=torch.float32, device=device)*torch.inf
	poses_min = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
	poses_min[:, 0, 0] = poses_min[:, 1, 1] = poses_min[:, 2, 2] = 1
	generated_images = torch.zeros(batch_size, int(np.sqrt(npix)), int(np.sqrt(npix)),  dtype=torch.float32, device=device)
	for batch_poses in tqdm(enumerate(poses)):
		batch_poses = batch_poses[None, :, :].repeat(batch_size, 1)
		all_wigner = wigner_calculator.compute_wigner_D(l_max, batch_poses, device)
		#all_wigner = utils.compute_wigner_D(l_max, batch_poses, device)
		rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
		predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate, rotated_spherical_harmonics, circular_mask.mask, radius_indexes, device)
		if use_ctf:
		    batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
		else:
		    batch_predicted_images = predicted_images

		#Instead of averaging over the batch dimension, we keep it to know which pose minimizes the reconstruction error for each pose.
		losses = torch.mean((batch_translated_images_hartley - batch_predicted_image.flatten(start_dim=1, end_dim=2))**2, dim=-1)
		poses_min[loss < reconstruction_errors] = batch_poses[loss < reconstruction_errors]
		reconstruction_errors[loss < reconstruction_errors] = loss[loss < reconstruction_errors]
		argmin_images[loss < reconstruction_errors] = batch_predicted_image[loss < reconstruction_errors]
		return poses_min, reconstruction_errors, argmin_images