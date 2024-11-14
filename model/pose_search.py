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
	top_k_val = loss.topk(max_poses, dim=-1, largest=False, sorted=True)[0][:, -1, None] # [batch_size, 1]
	max_top_k_values = max_poses
	#The following condition handles the possibles ties. If there is any tie on any batch, we increase the top k value we retain to the max value of top k for the batch.
	max_top_k_values = torch.max(torch.sum(loss <= top_k_val, dim=-1))
	if max_top_k_values > 8:
		top_k_val = loss.topk(max_top_k_values, dim=-1, largest=False, sorted=True)[0][:, -1, None] # [batch_size, 1]

	batch_number, rotation_to_keep = (loss <= top_k_val).nonzero(as_tuple=True) # [batch_size*max_poses,], [batch_size*max_poses,]
	print("NUmber of k lowest values:", torch.sum(loss <= top_k_val, dim=-1))
	return batch_number, rotation_to_keep, max_top_k_values


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
		self.base_grid_rot_mat = torch.tensor(scipy.spatial.transform.Rotation.from_quat(base_grid["quat"][:, [1, 2, 3, 0]]).as_matrix(), dtype=torch.float32, device=device)
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

	@timing
	def subdivide(self, quat, q_ind, res):
		"""
		Subdivide the SO(3) grid in the points that we keep.
		:param quat: torch.tensor(nq, 4)
		:param ind: torch.tensor(nq, 2) with s2 indexes in (nq, 0) and s1 indexes in (nq, 1)
		:param res: integer, current resolution.
		return torch.tensor(nq, 8, 4) of quaternions in the new points, torch.tensor(nq, 8, 2) of the indexes of the points on SO(3)
		"""
		nq = quat.shape[0]
		new_s2_coordinates, new_s2_ind = healpy_grid.get_s2_neighbor_tensor(q_ind[:, 0], res)
		theta, phi = new_s2_coordinates
		psi, new_s1_ind = healpy_grid.get_s1_neighbor_tensor(q_ind[:, 1], res)
		quat_new = healpy_grid.hopf_to_quat_tensor(
		np.repeat(theta[..., None], psi.shape[-1], axis=-1).reshape(nq, -1),
		np.repeat(phi[..., None], psi.shape[-1], axis=-1).reshape(nq, -1),
		np.repeat(psi[:, None], theta.shape[-1], axis=-2).reshape(nq, -1)
		) # nq, 16, 4
		ind_new = np.concatenate([
		np.repeat(new_s2_ind[..., None], psi.shape[-1], axis=-1).reshape(nq, -1)[..., None],
		np.repeat(new_s1_ind[:, None], theta.shape[-1], axis=-2).reshape(nq, -1)[..., None]
		], -1)  # nq, 16, 2

		quat_new = torch.tensor(quat_new).to(self.device)
		dists = torch.minimum(
		torch.sum((quat_new - quat[:, None, :]) ** 2, dim=-1),
		torch.sum((quat_new + quat[:, None, :]) ** 2, dim=-1)
		)  # nq, 16
		ii = torch.argsort(dists, dim=-1)[:, :self.max_poses].cpu()  #For each of the nq points, find the indices of the eight nearest points
		quat_out = quat_new[torch.arange(nq)[..., None], ii] #Get the corresponding quaternions.
		ind_out = ind_new[torch.arange(nq)[..., None], ii] #Get the corresponding SO(3) indexes.
		return quat_out, ind_out

	@timing
	def evaluate_images(self, alms_per_coordinates, all_wigner, spherical_harmomics, k, n_so3_points):
		"""
		Evaluates the images for a set of rotation and a given alms
		:param alms_per_coordinates: torch.tensor(batch_size, N_pixels_in_bigger_mask, (lmax + 1)**2) of alms coefficient for each sample in batch
		:param all_wigner: list of torch.tensor(n_so3_points, 2l+1, 2l+1) of wigner matrices, containing, for each of the rotation in the pose search, the wigner D matrix for l.
		:param spherical_harmonics: torch.tensor(N_pixels_in_bigger_mask, (lmax+1)**2)
		:param k: integer, current resolution of the masking in Fourier space.
		:param n_so3_points: integer, number of points in SO(3) that we are evaluating, so after the first iteration it is batch_size*self.max_poses*8
		return torch.tensor(batch_size*n_so3_points, n_pix**2) of predicted images
		"""
		batch_size = alms_per_coordinate.shape[0]
		###########       I can surely make the next line faster by conidering only k_min instead of the circular mask defined in utils !!!!! ######
		print("ON GPU WIGNER ?", all_wigner[1].device)
		print("ON GPU SPH ?", spherical_harmonics[1].device)
		print("ON GPU ALMs ?", alms_per_coordinate.device)
		rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max) # [N_points_base_grid, N_pixels_in_mask, (lmax+1)**2] Get the rotated sph for each of the base grid points
		if k == self.kmin:
			rotated_spherical_harmonics = rotated_spherical_harmonics.repeat(batch_size, 1, 1) # [batch_size*N_points_base_grid, N_pixels_in_mask, (lmax+1)**2]

		mask_freq = self.mask.get_mask(k) #We define a new mask corresponding to kmin
		mask_freq_in_circular_mask = mask_freq[self.circular_mask == 1] #We get the elements of the mask of kmin that should be included in the mask defined at the start of the run
		radius_indexes, unique_radiuses = utils.get_radius_indexes(self.frequencies.freqs, mask_freq, self.device) #We get all the unique radiuses in kmin and their indexes
		batch_predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate[:, mask_freq_in_circular_mask == 1].repeat_interleave(n_so3_points, dim=0), 
						rotated_spherical_harmonics[:, mask_freq_in_circular_mask == 1], mask_freq, radius_indexes, device) # [batch_size*N_points_base_grid, npix, npix] of predicted images


		return batch_predicted_images

	@timing
	def get_indices_to_keep(self, true_images, batch_predicted_images, grid_quat, grid_idx, k):
		"""
		:param true_images: torch.tensor(batch_size, n_pix**2)
		:param batch_predicted_images: torch.tensor(batch_size*self.max_poses*8, n_pix**2)
		:param grid_quat: torch.tensor(batch_size*self.max_poses, 8, 4)
		:param grid_quat: torch.tensor(batch_size*self.max_poses, 8, 2)
		:param k: integer, radius of the mask we apply in Fourier coordinates.
		"""
		batch_size = true_images.shape[0]
		n_so3_points = int(batch_predicted_images.shape[0]/batch_size)
		mask_freq = self.mask.get_mask(k) #
		batch_predicted_images = batch_predicted_images.flatten(start_dim=1, end_dim=2)
		#########      BE CAREFUL I AM NOT APPLYING ANY CTF HERE !!!!!!!!!!! ##########
		losses = torch.mean((true_images[:, mask_freq==1].repeat_interleave(n_so3_points, dim=0) - batch_predicted_images[:, mask_freq==1])**2, dim=-1).reshape(batch_size, n_so3_points)
		batch_number, rotation_to_keep, max_poses = keep_matrix_simpler(losses, self.max_poses) # [batch_size*self.max_poses*8,], [batch_size*self.max_poses*8,]
		keep_quat = grid_quat.reshape(batch_size, -1, 4)[batch_number, rotation_to_keep] #tensor of shape [batch_size*n_so3_points, 4]
		keep_ind = grid_idx.reshape(batch_size, -1, 2)[batch_number, rotation_to_keep] # tensor of shape [batch_size*n_so3_points, 2]
		return keep_ind, keep_quat, rotation_to_keep, losses, max_poses



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
		all_wigner = self.all_wigner_base # list of tensot (n_so3_points_in_base_grid, 2l+1, 2l+1)
		for n_iter in range(0, self.total_iter+1):
			start = time()
			resolution = self.base_resol + n_iter
			k = self.get_frequency_limit(n_iter)
			print(f"N_iter: {n_iter}, resolution Fourier {k}, Resolution SO(3) grid {resolution}")
			batch_predicted_images = self.evaluate_images(alms_per_coordinate, all_wigner, spherical_harmonics, k, n_so3_points)
			#Be careful: we keep the eight lowest reconstruction error, but we split these into 8 new points (the s2 grid pixels are divided in 4 and the s1 grid pixels are divided in 2)
			keep_ind, keep_quat, rotation_to_keep, losses, max_poses = self.get_indices_to_keep(true_images, batch_predicted_images, base_grid_q, base_grid_idx, k)
			base_grid_q, base_grid_idx = self.subdivide(keep_quat, keep_ind, resolution) # tensor of shape [batch_size*self.max_poses, 8, 4] and [[batch_size*self.max_poses, 8, 2] 
			base_grid_rotmat = quaternion_to_matrix(base_grid_q).reshape(-1, 3, 3)
			all_wigner = precompute_wigner_D(self.wigner_calculator, base_grid_rotmat, self.l_max, self.device)
			#Note that it is necessary to use the redefined max_poses: of there are ties in the top k lowest reconstruction errors, we increase the number of poses kept.
			n_so3_points = max_poses*8
			end = time()
			print("Total time one iteration", end-start)
			print(f"MIN LOSS AT ITER {n_iter}:", losses.min(1))

		return rotation_to_keep.cpu().numpy(), losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
kmin = 12
kmax = 94
l_max = 5
N_images = 10
elts = [50, 313, 200, 3, 5, 500, 315]
elts = np.random.randint(low = 0, high=576, size=(N_images, )).tolist()
frequencies = grid.Grid(190, 1.0, device)
wigner_calculator = WignerD(l_max, device=device)
#with open("data/dataset/1_resol.json", "r") as f:
#	base_grid = json.load(f)

 
base_grid = {"quat":np.load("data/dataset/1_resol_quat.npy"), "ind":np.load("data/dataset/1_resol_ind.npy"), "resol":1}
circular_mask = Mask(190, 1.0, radius = 95)
#Getting the spherical harmonics object.
sh = sct.SphericalHarmonics(l_max=l_max, normalized=True)
#Computing the spherical harmonics.
spherical_harmonics = utils.get_real_spherical_harmonics(frequencies.freqs[circular_mask.mask ==1], sh, device, l_max)
#Sampling alms to create the images
alms_per_coordinate = torch.randn(N_images, 190**2, (l_max+1)**2, dtype=torch.float32, device=device)
#alms_per_coordinate[1] = alms_per_coordinate[0]
#Setting the frequencies outside the small mask to 0
alms_per_coordinate[:, circular_mask.get_mask(kmin) != 1] = 0
#Keeping only the big mask
alms_per_coordinate = alms_per_coordinate[:, circular_mask.mask ==1] 
#Setting the alms outside of the smallest mask to 0, so we can limit ourselves to 1 iteration.
radius_indexes, unique_radiuses = utils.get_radius_indexes(frequencies.freqs, circular_mask.mask , device)
N_unique_radiuses = len(unique_radiuses)
#Get the quaternions of some poses
quat_poses = base_grid["quat"][elts]
#Get the corresponding indices
indices_poses = base_grid["ind"][elts]
#Defining the pose search object
pose_search = PoseSearch(kmin, kmax, wigner_calculator, base_grid, circular_mask = circular_mask.mask, frequencies=frequencies, l_max=l_max, npix=190, apix=1.0, total_iter=5, max_poses = 8, n_neighbors=8, device=device)


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



rotation_to_keep, losses = pose_search.search_new(alms_per_coordinate, true_images, spherical_harmonics, l_max, device, ctf=None, indexes=None)

#print("Rotation to keep", rotation_to_keep)
#print("Poses Quaternions", quat_poses)
#print(losses[0, 50], losses[0, 255])
#print(losses.topk(2, dim=-1, largest=False, sorted=True)[1])

print("Max loss", torch.max(losses.min(1)[0]))
print(rotation_to_keep)
print("Max distances between true and recover index on SO(3)", np.max(np.abs(np.array(rotation_to_keep) - np.array(elts))))





