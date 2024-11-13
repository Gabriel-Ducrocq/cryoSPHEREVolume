from tqdm import tqdm
from time import time
import utils
#from model import utils
import loss
import renderer
from grid import Mask
import torch
#import model
import scipy
import sphericart as sct
import numpy as np
import grid
from time import time
from grid import Grid, rotate_grid
from decorators import timing
from wignerD import WignerD
from pytorch3d.transforms import quaternion_to_matrix
import healpy_grid



@timing
def precompute_wigner_D(wigner_calculator, poses, l_max, device="cpu"):
	"""
	This function pre compute the wigner D matrices for all the given poses, since it seems to be the most expensive step in the loop.
	:param poses: torch.tensor(N_poses, 3, 3) rotation matrices corresponding to the poses.
	"""
	return wigner_calculator.compute_wigner_D(l_max, poses.to(device), device)



def get_neighbor_so3(
        quat,
        q_ind,
        res,
        device
):
    """
    nq = batch_size*max_poses
    quat: [nq, 4]
    q_ind: [nq, 2], np.array
    cur_res: int

    output: [nq, 8, 4], [nq, 8, 2] (np.array)
    """
    return healpy_grid.get_neighbor_tensor(quat, q_ind, res, device)



def keep_matrix(
        loss,
        batch_size,
        max_poses
	):
    """
    loss: [batch_size, q]: tensor of losses for each rotation.

    output: 3 * [batch_size, max_poses]: bool tensor of rotations to keep, along with the best translation for each.
    """
    shape = loss.shape
    assert len(shape) == 2
    #Gets the top k minimal losses over the rotations for each sample
    flat_idx = loss.topk(max_poses, dim=-1, largest=False, sorted=True)[1]
    # Since we are going to flatten the entire tensor, we need a way to keep the batch index of all k minimum losses per batch
    #For each batch, the previous line give us the idx relative to each sample. We just add the batch index times the number of rotations to it so that we obtain
    #the index once the big tensor is flattened.
    flat_idx += (
        torch.arange(batch_size, device=loss.device).unsqueeze(1) * loss.shape[1]
    )
    #Flatten all the indices. It contains the indices of all the rotations we want to take, expressed in the flatten batch*N_rotations format.
    flat_idx = flat_idx.reshape(-1)

    #We create an empty tensor to receive all the poses.
    keep_idx = torch.empty(
        len(shape), batch_size * max_poses, dtype=torch.long, device=loss.device
    )
    # For each index to keep, dividing it (Euclidean way) by max_poses will give the batch number it corresponds to
    keep_idx[0] = torch.div(flat_idx, shape[1], rounding_mode='trunc')
    # For each index to keep, getting its rest in the division by max_poses will the pose number it corresponds to.
    keep_idx[1] = flat_idx % shape[1]

    return keep_idx

def keep_matrix_simpler(loss, max_poses):
	"""
	THis function is the same as above, but simpler. However, the indexes are not sorted from lower error to higher error, contrary to above function !
	:param loss: torch.tensor(batch_size, q)
	return torch.tensor(batch_size*max_poses), torch.tensor(batch_size*max_poses)
	"""
	top_k_val = loss.topk(max_poses, dim=-1, largest=False, sorted=True)[0][:, -1, None]
	batch_number, rotation_to_keep = (loss <= top_k_val).nonzero(as_tuple=True)
	return batch_number, rotation_to_keep


class PoseSearch:
	def __init__(self, kmin, kmax, wigner_calculator, base_grid, npix, apix, circular_mask, frequencies, total_iter=5, max_poses = 8, n_neighbors=8, l_max = 2, device="cpu"):
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
		self.l_max = l_max
		self.total_iter = total_iter
		self.mask = Mask(npix, apix)
		self._so3_neighbor_cache = {}
		self.base_grid_rot_mat = torch.tensor(scipy.spatial.transform.Rotation.from_quat(base_grid["quat"][:, [1, 2, 3, 0]]).as_matrix(), dtype=torch.float32)
		self.all_wigner_base = precompute_wigner_D(wigner_calculator, self.base_grid_rot_mat, self.l_max)
		self.base_quaternions = base_grid["quat"]
		self.max_poses = max_poses
		self.neighbors = n_neighbors
		self.wigner_calculator = wigner_calculator
		self.device=device
		self.base_resol = base_grid["resol"]
		self.npix = npix
		self.apix = apix
		self.circular_mask = circular_mask
		self.frequencies = frequencies
		self._so3_neighbor_cache = {}
		self.base_grid_idx = base_grid["ind"]

	def get_frequency_limit(self, n_iter):
		return min(self.kmin + int((n_iter/self.total_iter) * (self.kmax - self.kmin)), self.npix//2)

	def apply_mask(self, images, k):
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
		    self._so3_neighbor_cache[key] = healpy_grid.get_neighbor(quat, s2i, s1i, res)
		# FIXME: will this cache get too big? maybe don't do it when res is too
		return self._so3_neighbor_cache[key]


	def get_neighbor_so3(quat, q_ind, res, device):
		"""
		quat: [nq, 4]
		q_ind: [nq, 2], np.array
		cur_res: int

		output: [nq, 8, 4], [nq, 8, 2] (np.array)
		"""
		return healpy_grid.get_neighbor_tensor(quat, q_ind, res, device)

	def subdivide(self,
	        quat,
	        q_ind,
	        cur_res,
	        device
	):
	    """
	    Subdivides poses for next resolution level.
		nq = batch_size*max_poses
	    quat: [nq, 4]: quaternions
	    q_ind: [nq, 2]: np.array, index of current S2xS1 grid
	    cur_res: int: Current resolution level

	    output:
	        quat  [nq, 8, 4]
	        q_ind [nq, 8, 2]
	        rot   [nq * 8, 3, 3]
	    """
	    quat, q_ind = get_neighbor_so3(quat, q_ind, cur_res, device)
	    rot = lie_tools.quat_to_rotmat(quat.reshape(-1, 4))
	    return quat, q_ind, rot


	def replace_wigner(self, argmin_wigner, wigner, losses_lb, reconstruction_errors, l_max):
		"""
		This functions takes replaces the batch_dimensions of argmin_wigner with the one of wigner, only on the samples where losses_lb < reconstruction_errors
		:param reconstruction_errors: torch.tensor(batch_size, ) of the lowest error so far.
		"""
		for ell in range((l_max+1)**2):
			argmin_wigner[ell][losses_lb < reconstruction_errors] = wigner[ell][losses_lb < reconstruction_errors]


	def search(self, alms_per_coordinate, true_images, spherical_harmonics, l_max, device, ctf, indexes):
		"""
		Perform the pose search with frequency marching
		:param alms_per_coordinate: torch.tensor(batch_size, N_pixels_in_mask, (l_max+1)**2)
		:param true_images: torch.tensor(batch_size, side_shape**2) of true, translated images, expressed in Hartley space.
		"""
		batch_size = true_images.shape[0]
		n_so3_points = len(self.all_wigner_base[0])
		alms_per_coordinate = alms_per_coordinate.repeat_interleave(n_so3_points, dim=0)
		repeat_indexes = torch.arange(batch_size).repeat_interleave(n_so3_points)
		#Note that all_wigner is a list containing tensor of size (batch_size x n_so3_points, 2l+1, 2l+1)
		all_wigner = self.all_wigner_base
		#Since I know we keep only 8 rotations, I can store all the max_pose*n_neighbors current quaternions per sample in a tensor.
		all_quaternions = torch.zeros(batch_size, n_so3_points, 4, dtype=torch.float32, device=self.device)
		all_quaternions = torch.zeros(batch_size*8, 4, dtype=torch.float32, device=self.device)
		#Same, I can keep in memory the indices corresponding to each rotation for each sample
		all_indices = np.zeros((batch_size, n_so3_points, 2), dtype=int)
		for n_it in range(1, self.total_iter+1):
			print("Iteration number:", n_it)
			resol = self.base_resol + (n_it -1) #n_it start at 1 but we want our resol to start at 1
			k = self.get_frequency_limit(n_it)
			mask_freq = self.mask.get_mask(k)
			mask_freq2 = Mask(self.npix, self.apix, radius=k)
			radius_indexes, unique_radiuses = utils.get_radius_indexes(self.frequencies.freqs, mask_freq2, self.device)
			#We create a tensor for storing all the losses for all batches and all poses.
			all_losses_lb = torch.ones(batch_size, n_so3_points, dtype=torch.float32, device=self.device)*torch.inf
			#We make a rotation of the spherical harmomics
			rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
			rotated_spherical_harmonics = rotated_spherical_harmonics.repeat(batch_size, 1, 1)
			#We want to know which component to keep in the frame of the circular mask we apply, given that our new mask is smaller.
			mask_freq_in_circular_mask = mask_freq[self.circular_mask == 1]
			#We only synthesize the frequencies within our new, smaller mask
			predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate[:, mask_freq_in_circular_mask == 1], 
															rotated_spherical_harmonics[:, mask_freq_in_circular_mask == 1], mask_freq, radius_indexes, device)
			if ctf is not None:
				#We have to repat the indexes for the ctf, because we have the same image for n_so3_points repeated.
				indexes = indexes.repeat(n_so3_points, 0)
				batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
			else:
				batch_predicted_images = predicted_images

			batch_predicted_images = batch_predicted_images.flatten(start_dim=1, end_dim=2)
			losses = torch.mean((true_images[:, mask_freq==1][repeat_indexes] - batch_predicted_images[:, mask_freq==1])**2, dim=-1).reshape(batch_size, n_so3_points)
			#batch_idx and poses_idx are (batch_size*self.max_poses)
			batch_idx, poses_idx = keep_matrix_simpler(losses, self.max_poses)
			#quat is (batch_size*self.max_poses, 4)
			quat = all_quaternions[batch_idx, poses_idx]
			#idx is (batch_size*self.max_poses, 2) tensor of indexes on SO(3)
			idx = all_indices[batch_idx, poses_idx]
			all_quaternions, all_indices, all_wigner = self.subdivide(all_quaternions, idx, resol, device)
			n_so3_points = self.max_poses*self.neighbors

		return all_indices


	def search_new(self, alms_per_coordinate, true_images, spherical_harmonics, l_max, device, ctf, indexes):
		"""
		Perform the pose search with frequency marching
		:param alms_per_coordinate: torch.tensor(batch_size, N_pixels_in_mask, (l_max+1)**2)
		:param true_images: torch.tensor(batch_size, side_shape**2) of true, translated images, expressed in Hartley space.
		"""
		batch_size = true_images.shape[0]
		n_so3_points = len(self.all_wigner_base[0]) #number of points in the base grid of SO(3)
		base_grid_q = self.base_quaternions[None, :, :].repeat(batch_size, axis=0) #shape [batch_size, N_points_base_grid, 4] np.array
		base_grid_idx = self.base_grid_idx[None, :, :].repeat(batch_size, axis=0) #shape [batch_size, N_points_base_grid, 2] for coordinates on s2 and s1
		print("BASE GRID", self.base_grid_idx)
		###########       I can surely make the next line faster by conidering only k_min instead of the circular mask defined in utils !!!!! ######
		rotated_spherical_harmonics = utils.apply_wigner_D(self.all_wigner_base, spherical_harmonics, l_max) # [N_points_base_grid, N_pixels_in_mask, (lmax+1)**2] Get the rotated sph for each of the base grid points
		rotated_spherical_harmonics = rotated_spherical_harmonics.repeat(batch_size, 1, 1) # [batch_size*N_points_base_grid, N_pixels_in_mask, (lmax+1)**2]
		mask_freq = self.mask.get_mask(self.kmin) #We define a new mask corresponding to kmin
		mask_freq_in_circular_mask = mask_freq[self.circular_mask == 1] #We get the elements of the mask of kmin that should be included in the mask defined at the start of the run
		radius_indexes, unique_radiuses = utils.get_radius_indexes(self.frequencies.freqs, mask_freq, self.device) #We get all the unique radiuses in kmin and their indexes
		batch_predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate[:, mask_freq_in_circular_mask == 1].repeat_interleave(n_so3_points, dim=0), 
							rotated_spherical_harmonics[:, mask_freq_in_circular_mask == 1], mask_freq, radius_indexes, device) # [batch_size*N_points_base_grid, npix, npix] of predicted images

		#########      BE CAREFUL I AM NOT APPLYING ANY CTF HERE !!!!!!!!!!! ##########
		batch_predicted_images = batch_predicted_images.flatten(start_dim=1, end_dim=2)
		losses = torch.mean((true_images[:, mask_freq==1].repeat_interleave(n_so3_points, dim=0) - batch_predicted_images[:, mask_freq==1])**2, dim=-1).reshape(batch_size, n_so3_points)
		batch_number, rotation_to_keep = keep_matrix_simpler(losses, self.max_poses)
		return batch_number, rotation_to_keep.cpu().numpy(), losses



import matplotlib.pyplot as plt
torch.manual_seed(15)
kmin = 12
kmax = 35
l_max = 2
N_images = 10
elts = [50, 313, 200, 3, 5, 500, 315]
elts = np.random.randint(low = 0, high=576, size=(N_images, )).tolist()
device="cpu"
frequencies = grid.Grid(190, 1.0, "cpu")
wigner_calculator = WignerD(l_max, device="cpu")
#with open("data/dataset/1_resol.json", "r") as f:
#	base_grid = json.load(f)

 
base_grid = {"quat":np.load("data/dataset/1_resol_quat.npy"), "ind":np.load("data/dataset/1_resol_ind.npy"), "resol":1}
circular_mask = Mask(190, 1.0, radius = 95)
#Getting the spherical harmonics object.
sh = sct.SphericalHarmonics(l_max=l_max, normalized=True)
#Computing the spherical harmonics.
spherical_harmonics = utils.get_real_spherical_harmonics(frequencies.freqs[circular_mask.mask ==1], sh, "cpu", l_max)
#Sampling alms to create the images
alms_per_coordinate = torch.randn(N_images, 190**2, (l_max+1)**2, dtype=torch.float32)
#alms_per_coordinate[1] = alms_per_coordinate[0]
#Setting the frequencies outside the small mask to 0
alms_per_coordinate[:, circular_mask.get_mask(kmin) != 1] = 0
#Keeping only the big mask
alms_per_coordinate = alms_per_coordinate[:, circular_mask.mask ==1] 
#Setting the alms outside of the smallest mask to 0, so we can limit ourselves to 1 iteration.
radius_indexes, unique_radiuses = utils.get_radius_indexes(frequencies.freqs, circular_mask.mask , "cpu")
N_unique_radiuses = len(unique_radiuses)
#Get the quaternions of some poses
quat_poses = base_grid["quat"][elts]
#Get the corresponding indices
indices_poses = base_grid["ind"][elts]
#Defining the pose search object
pose_search = PoseSearch(kmin, kmax, wigner_calculator, base_grid, circular_mask = circular_mask.mask, frequencies=frequencies, l_max=l_max, npix=190, apix=1.0, total_iter=1, max_poses = 1, n_neighbors=8)


if False:
	########    OLD WAY ########
	#Get the rotation matrices of these poses
	rot_mat_poses = torch.tensor(scipy.spatial.transform.Rotation.from_quat(quat_poses[:, [1, 2, 3, 0]]).as_matrix(), dtype=torch.float32)
	#Compute the corresponding Wigner matrices
	wigner_poses = precompute_wigner_D(wigner_calculator, rot_mat_poses, l_max)
	#Rotate the spherical harmonics
	poses_spherical_harmonics = utils.apply_wigner_D(wigner_poses, spherical_harmonics, l_max)
	true_images = utils.spherical_synthesis_hartley(alms_per_coordinate, poses_spherical_harmonics, circular_mask.mask, radius_indexes, device)
	#true_images_numpy = true_images.numpy()
	#plt.imshow(true_images_numpy[0])
	#plt.show()
	#print("DIff", np.max(true_images_numpy[0] - true_images_numpy[1]))
	true_images = true_images.flatten(start_dim=-2, end_dim=-1)

else:
	######    NEW WAY #######
	print(pose_search.base_grid_rot_mat[elts].shape)
	#rot_mat_poses = torch.tensor(scipy.spatial.transform.Rotation.from_quat(pose_search.base_grid_rot_mat[elts][:, [1, 2, 3, 0]]).as_matrix(), dtype=torch.float32)
	rot_mat_poses = pose_search.base_grid_rot_mat[elts]
	wigner_poses = precompute_wigner_D(wigner_calculator, rot_mat_poses, l_max)
	poses_spherical_harmonics = utils.apply_wigner_D(wigner_poses, spherical_harmonics, l_max)
	true_images = utils.spherical_synthesis_hartley(alms_per_coordinate, poses_spherical_harmonics, circular_mask.mask, radius_indexes, device)
	true_images = true_images.flatten(start_dim=-2, end_dim=-1)



batch_number, rotation_to_keep, losses = pose_search.search_new(alms_per_coordinate, true_images, spherical_harmonics, l_max, device, ctf=None, indexes=None)

print(batch_number)
#print("Rotation to keep", rotation_to_keep)
#print("Poses Quaternions", quat_poses)
#print(losses[0, 50], losses[0, 255])
#print(losses.topk(2, dim=-1, largest=False, sorted=True)[1])

print("Max loss", torch.max(losses.min(1)[0]))
print("Max distances between true and recover index on SO(3)", np.max(np.abs(np.array(rotation_to_keep) - np.array(elts))))





