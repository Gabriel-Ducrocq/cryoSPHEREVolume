import torch
import sphericart as sct
import wandb
import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
from dataset import ImageDataSet
from renderer import SpatialGridTranslate
from vae import VAE
from mlp import MLP
from ctf import CTF
import yaml
from wignerD import WignerD
from grid import Grid, Mask
import numpy as np
#from astropy.coordinates import cartesian_to_spherical
import starfile
import e3nn
from time import time
import pytorch3d


def fourier2d_to_primal(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return: torch.tensor(batch_size, N_pix, N_pix) images in real space
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r

def get_radius_indexes(freqs, circular_mask, device):
    """
    Link the index of the unique indexes to the corresponding frequencies
    :param freqs: torch.tensor(side_hape**2, 3)
    :param circular_mask: object of class Mask.
    :return: torch.tensor(side_shape**2)
    """
    #Computes the radius in Fourier space:
    radius = torch.sqrt(torch.sum(freqs ** 2, axis=-1))
    #Get the radius within the mask.
    radius_within_mask = radius[circular_mask.mask==1]
    #Get the unique radiuses in the images within the mask
    unique_radius = torch.unique(radius_within_mask, sorted=True)
    #Creates one index per unique radius
    unique_indexes = torch.linspace(0, len(unique_radius)-1, len(unique_radius), dtype=torch.int, device=device)
    #Maps each unique radious to its index
    rad_and_ind = torch.stack([unique_radius, unique_indexes], dim=-1)
    #For each non unique radius within mask, get its index in the unique radius. This way, we know that different Fourier frequencies with same radiuses have the same entry in the vae.
    indexes = torch.stack([rad_and_ind[rad_and_ind[:, 0] == rad, 1] for rad in radius_within_mask], dim=0)
    return indexes.to(torch.int32)[:, 0], unique_radius

def parse_yaml(path):
    """
    Parse the yaml file to get the setting for the run
    :param path: str, path to the yaml file
    :return: settings for the run
    """
    with open(path, "r") as file:
        experiment_settings = yaml.safe_load(file)

    with open(experiment_settings["image_yaml"], "r") as file:
        image_settings = yaml.safe_load(file)

    if experiment_settings["device"] == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    use_ctf = experiment_settings["use_ctf"]
    l_max = experiment_settings["l_max"]
    particles_path = experiment_settings["particles_path"]
    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    apix_downsize = Npix * apix /Npix_downsize

    circular_mask = Mask(Npix_downsize, apix_downsize)

    frequencies = Grid(Npix_downsize, apix_downsize, device)
    radius_indexes, unique_radiuses = get_radius_indexes(frequencies.freqs, circular_mask, device)
    N_unique_radiuses = len(unique_radiuses)

    encoder = MLP(Npix_downsize**2,
                  experiment_settings["latent_dimension"] * 2,
                  experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device,
                  latent_type="continuous")
    decoder = MLP(experiment_settings["latent_dimension"], N_unique_radiuses*(l_max+1)**2,
                  experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)

    vae = VAE(encoder, decoder, device, latent_dim=experiment_settings["latent_dimension"], lmax=l_max)
    vae.to(device)

    if experiment_settings["optimizer"]["name"] == "adam":
        optimizer = torch.optim.Adam(vae.parameters(), lr=experiment_settings["optimizer"]["learning_rate"])
    else:
        raise Exception("Optimizer must be Adam")


    particles_star = starfile.read(experiment_settings["star_file"])
    ctf_experiment = CTF.from_starfile(experiment_settings["star_file"], apix = apix_downsize, side_shape=Npix_downsize , device=device)
    dataset = ImageDataSet(apix, Npix, particles_star["particles"], particles_path, down_side_shape=Npix_downsize)
    #dataset = ImageDataSet(apix, Npix, particles_star, particles_path, down_side_shape=Npix_downsize)

    scheduler = None
    if "scheduler" in experiment_settings:
        milestones = experiment_settings["scheduler"]["milestones"]
        decay = experiment_settings["scheduler"]["decay"]
        print(f"Using MultiStepLR scheduler with milestones: {milestones} and decay factor {decay}.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)

    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]
    sh = sct.SphericalHarmonics(l_max=l_max, normalized=True)
    spherical_harmonics = get_real_spherical_harmonics(frequencies.freqs[circular_mask.mask ==1], sh, device, l_max)
    wigner_calculator = WignerD(l_max, device)
    image_translator = SpatialGridTranslate(D=Npix_downsize, device=device)


    return vae, optimizer, image_translator, dataset, N_epochs, batch_size, sh, unique_radiuses, radius_indexes, experiment_settings, device, \
    scheduler, frequencies.freqs, frequencies.freqs_volume, l_max, spherical_harmonics, wigner_calculator, ctf_experiment, use_ctf, circular_mask

def get_real_spherical_harmonics(coordinates, sphericart_obj, device, l_max):
    """
    Computes the real, cartesian spherical harmonics functions.
    :param coordinates: torch.tensor(N_freqs, 3)
    :param sphericart_obj: sphericat object for spherical harmonics computation, until a defined l_max, normalized or not
    :return: torch.tensor(N_freqs, (l_max+1)**2)
    """
    sh_values = torch.as_tensor(sphericart_obj.compute(coordinates.detach().cpu().numpy()), dtype=torch.float32, device=device)
    splitted_sh_values = []
    start = 0
    for l in range(l_max + 1):
        end = start + 2*l+1
        splitted_sh_values.append(sh_values[...,start:end])
        start = end

    return splitted_sh_values

def get_real_spherical_harmonics_e3nn(coordinates, l_max):
    """
    Computes the real, cartesian spherical harmonics functions.
    :param coordinates: torch.tensor(N_batch, N_freqs, 3) where N_freqs can be N_side_freq**3 if volume reconstruction
                        and N_side_freq**2 if image reconstruction
    :param sphericart_obj: sphericat object for spherical harmonics computation, until a defined l_max, normalized or not
    :return: torch.tensor(N_batch, N_freqs, (l_max+1)**2)
    """
    #### BE CAREFUL, REMOVING THE EXCHANGE OF COORDINATES !!!!!!!!
    coordinates = coordinates[:, [1, 2, 0]]
    all_sh_values = []
    for l in range(l_max+1):
        all_sh_values.append(e3nn.o3.spherical_harmonics(l=l, x=coordinates, normalize=True))

    return all_sh_values


def alm_from_radius_to_coordinate(alm, radiuses_index):
    """
    The alm that the network outputs are on per radius. We need to match the coordinate to the radiuses
    #:param alm: torch.tensor(N_batch, N_unique_radius, (l_max+1)**2)
    :param alm: torch.tensor(N_batch, N_unique_radius , (l_max+1)**2)
    :param radiuses_index: torch.tensor(N_pix**2) of alm index corresponding to the radius of that coordinate
    :return: torch.tensor(N_batch, N_pix**2, (l_max+1)**2)
    """
    return alm[:, radiuses_index, :]

def spherical_synthesis_hartley(alm_per_coord, spherical_harmonics, circular_mask, indexes, device):
    """
    Computes the Hartley transform through a spherical harmonics synthesis
    :param alm: torch.tensor(N_batch, side_shape**2, (l_max+1)**2)
    :param spherical_harmonics:torch.tensor(N_batch, side_shape**2, (l_max+1)**2)
    :param circular_mask: torch.tensor(N_pix**2), whether the frequencies are in the mask or not.
    :param radiuses_index: torch.tensor(side_shape**2) of alm index corresponding to the radius of that coordinate
    :param device: torch device on which to perform the computations.
    :return: torch.tensor(N_batch, side_shape, side_shape)
    """
    #Here, the frequencies (0,0) gave NaN for the (l_max+1)**2 coefficients, except l = 0. We replace directly with the
    #estimates provided by the neural net
    batch_size = alm_per_coord.shape[0]
    side_shape = int(np.sqrt(len(circular_mask)))
    spherical_harmonics[:, indexes ==0, :] = 0
    images_radius_0_nan = torch.einsum("b s l, b s l -> b s", alm_per_coord, spherical_harmonics)
    images_radius_0_nan[:, indexes == 0] = alm_per_coord[:, indexes == 0, 0]
    flat_images = torch.zeros(batch_size, side_shape**2, device=device)
    flat_images[:, circular_mask == 1] = images_radius_0_nan
    return flat_images.reshape(batch_size, side_shape, side_shape)


def hartley_to_fourier(image,device,  mu=None, std=None ):
    """
    Converts a Hartley image into a FFT image
    :param image: torch.tensor(batch_size, side_shape**2). Once reshaped to (side_shape, side_shape), the frequency 0 is at the center
    :return: torch.tensor(batch_size, side_shape, side_shape)
    """
    side_shape = int(np.sqrt(image.shape[1]))
    batch_size = image.shape[0]
    assert side_shape % 2 == 0, f"Image must have an even number of pixels. Currently has {side_shape} pixels"
    ## The image now has zero at the center and the lowest frequency at the top left.
    image_square = image.reshape(batch_size, side_shape, side_shape)
    if mu is not None and std is not None:
        image_square*=std
        image_square+=mu

    low_x = image_square[:, 0, :]
    low_y = image_square[:, :, 0]
    ## Since the Fourier transform is side_shape periodic, we can safely remove the lowest frequency and deal with it
    ## separately
    image_cropped = image_square[:, 1:, 1:]
    image_cropped_flipped = image_cropped.flip(dims=(1, 2))
    fourier_transform = torch.view_as_complex(torch.zeros((image_square.shape[0], image_square.shape[1], image_square.shape[2], 2), dtype=torch.float32, device=device))
    fourier_transform[:, 1:, 1:] = (image_cropped + image_cropped_flipped)/2 - 1j*(image_cropped - image_cropped_flipped)/2
    ### CHECK THIS PART !!!!!
    fourier_transform[:, 0, 1:] = (low_x[:, 1:] + torch.flip(low_x[:, 1:], dims=(0, )))/2 + 1j*(low_x[:, 1:] - torch.flip(low_x[:, 1:], dims=(0, )))/2
    fourier_transform[:, 1:, 0] = (low_y[:, 1:] + torch.flip(low_y[:, 1:], dims=(0, )))/2 + 1j*(low_y[:, 1:] - torch.flip(low_y[:, 1:], dims=(0, )))/2
    fourier_transform[:, 0, 0] = low_x[:, 0]
    return fourier_transform

def hartley_to_fourier_3d(volume, device):
    fourier_volume = torch.view_as_complex(torch.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 2), dtype=torch.float32, device=device))
    volume_cropped = volume[1:, 1:, 1:]
    volume_cropped_flipped = volume_cropped.flip(dims=(0, 1, 2))
    fourier_volume[1:, 1:, 1:] = (volume_cropped + volume_cropped_flipped)/2 + 1j*(volume_cropped - volume_cropped_flipped)/2
    fourier_volume[0, 1:, 1:] = (volume[0, 1:, 1:] + torch.flip(volume[0, 1:, 1:], dims=(0, 1)))/2 + 1j*(volume[0, 1:, 1:] - torch.flip(volume[0, 1:, 1:], dims=(0, 1)))/2
    fourier_volume[1:, 0, 1:] = (volume[1:, 0, 1:] + torch.flip(volume[1:, 0, 1:], dims=(0, 1))) / 2 + 1j * (volume[1:, 0, 1:] - torch.flip(volume[1:, 0, 1:], dims=(0, 1))) / 2
    fourier_volume[1:, 1:, 0] = (volume[1:, 1:, 0] + torch.flip(volume[1:, 1:, 0], dims=(0, 1))) / 2 + 1j * (volume[1:, 1:, 0] - torch.flip(volume[1:, 1:, 0], dims=(0, 1))) / 2
    fourier_volume[0, 0, 0] = volume[0, 0, 0]
    return volume




def fourier_to_hartley(fft_images):
    """
    Computes the Hartley transform of images in Fourier space.
    :param fft_images: torch.tensor(N_batch, N_freq, N_freq)
    return torch.tensor(N_batch, N_freq, N_freq)
    """
    return fft_images.real - fft_images.imag

def real_to_hartley(images):
    """
    Computes the Hartley transform of images in real space.
    :param images: torch.tensor(N_batch, N_pix, N_pix)
    return torch.tensor(N_batch, N_freq, N_freq)
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_proj = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])),
                                        dim=(-2, -1))
    return fourier_to_hartley(fourier_proj)

def hartley_to_real(images_hartley):
    """
    Computes the real images base on their Hartley transform.
    :param images: torch.tensor(N_batch, N_freq, N_freq)
    return torch.tensor(N_batch, N_pix, N_pix)
    """
    #Hartley transform is an involution, so hartley_transform(hartley_images) get the Fourier transform back
    images_fft = real_to_hartley(images_hartley)
    r = torch.fft.ifftshift(image_fft, dim=(-2, -1))
    images_real = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])),
                                        dim=(-2, -1)).real
    return images_real



def fourier_to_real(fft_images):
    """
    Compute the inverse fft to recover the image
    :param fft_images: torch.tensor(batch_size, side_shape, side_shape) of images in fourier space
    :return: torch.tensor(batch_size, side_shape, side_shape)
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(fft_images, dim=(-1, -2)))).real

#def hartley_to_real(images, device, mu=None, std=None):
#    """
#    Goes from Hartley space to real space
#    :param images: torch.tensor(batch_size, side_shape, side_shape) of images in Hartley space.
#    :param device: device, either gpu or cpu
#    :return: torch.tensor(batch_size, side_shape, side_shape) of images in real space.
#    """
#    fft_images = hartley_to_fourier(images, device, mu, std)
#    return fourier_to_real(fft_images)

def hartley_transform_3d(volume):
    """
    Performs Hartley transform of a 3d volumes
    :param volume: torch.tensor(batch_size, side_shape, side_shape, side_shape)
    :return: torch.tensor(batch_size, side_shape, side_shape, side_shape)
    """
    volume = torch.fft.ifftshift(volume, dim=(-3, -2, -1))
    fourier_volume = torch.fft.fftn(volume, dim=(-3, -2, -1))
    fourier_volume = torch.fft.fftshift(fourier_volume, dim=(-3, -2, -1))
    hartley_volume = fourier_volume.real - fourier_volume.imag
    return hartley_volume

def monitor_training(tracking_metrics, epoch, experiment_settings, vae, optimizer, device=None, true_images=None, predicted_images=None, real_image=None,
                     images_mean = None, images_std = None):
    """
    Monitors the training process through wandb and saving masks and models
    :param mask:
    :param tracking_metrics:
    :param epoch:
    :param experiment_settings:
    :param vae:
    :return:
    """
    side_shape = int(np.sqrt(predicted_images.shape[1]))
    batch_size = predicted_images.shape[0]
    predicted_images = predicted_images.reshape(batch_size, side_shape, side_shape)
    predicted_images*=(images_std + 1e-15)
    predicted_images+= images_mean
    real_predicted_image = real_to_hartley(predicted_images[:1])

    true_images = true_images.reshape(batch_size, side_shape, side_shape)
    real_image_again = real_to_hartley(true_images[:1])

    wandb.log({key: np.mean(val) for key, val in tracking_metrics.items()})
    wandb.log({"epoch": epoch})
    wandb.log({"lr":optimizer.param_groups[0]['lr']})
    real_image_again_wandb = wandb.Image(real_image_again[0].detach().cpu().numpy()[:, :, None], caption="Round trip real to Hartley")
    true_image_ony_real_wandb = wandb.Image(real_image[0].detach().cpu().numpy()[:, :, None],
                                         caption="Original image")
    pred_im = real_predicted_image[0].detach().cpu().numpy()[:, :, None]
    print("PRED IM", pred_im.shape)
    predicted_image_wandb = wandb.Image(pred_im,
                                         caption="Predicted images")
    wandb.log({"Images/true_image": real_image_again_wandb})
    wandb.log({"Images/true_image_ony_real": true_image_ony_real_wandb})
    wandb.log({"Images/predicted_image": predicted_image_wandb})
    torch.save(vae, experiment_settings["folder_path"] + "models_main/full_model" + str(epoch))



def convert_spher_cartesian(lat, long, r):
    return np.array((r*np.sin(lat)*np.cos(long), r*np.sin(lat)*np.sin(long), r*np.cos(lat)))[None, :]

def compute_wigner_D(l_max, R, device):
    """

    :param l_max: int, l_max for the spherical harmonics
    :param R: torch.tensor(N_batch, 3, 3)
    :return:
    """
    r = []
    alpha, beta, gamma = e3nn.o3.matrix_to_angles(R[:, [1, 2, 0], :][:, :, [1, 2, 0]])
    alpha = alpha.detach().cpu()
    beta = beta.detach().cpu()
    gamma = gamma.detach().cpu()
    start = time()
    for l in range(l_max+1):
        r_inter = e3nn.o3.wigner_D(l, alpha, beta, gamma)
        r.append(r_inter.to(device))

    end = time()
    print("Inside Wigner time", end - start)
    return r

def apply_wigner_D(wigner_matrices, spherical_harmonics, l_max):
    """

    :param lmax:
    :param wigner_matrices:
    :param spherical_harmonics: (batch_size, s**2, (l_max + 1)**2)
    :return:
    """
    res = []
    for l in range(l_max+1):
        print("INSIDE")
        print(wigner_matrices[l].shape)
        print(spherical_harmonics[l].shape)
        r = torch.einsum("b e l , s l-> b s e", wigner_matrices[l], spherical_harmonics[l])
        res.append(r)

    res = torch.cat(res, dim=-1)
    return res


"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
l_max = 3
#sh = sct.SphericalHarmonics(l_max=l_max, normalized=True)
coordinates = torch.randn((1, 3), dtype=torch.float32)
start_old = time()
spherical_harmonics = get_real_spherical_harmonics_e3nn(coordinates, l_max)
sh = sct.SphericalHarmonics(l_max=l_max, normalized=True)
#print("SPHERICAL HARMONICS E3NN SHAPE", spherical_harmonics.shape)
#end_old = time()
#print("Old version", end_old - start_old)
#start_old = time()
R,Res = torch.linalg.qr(torch.rand(1, 3, 3))
### We place ourselves in the convention (x, y, z), with a coordinate v and rotation matrix R
### To get the real spherical harmonics, we need the convention (y, z, x) to feed into e3nn
## We can permute v to get v_prime = Uv where U is a permutation matrix
## But a rotation of v by R is not a rotation of v_prime by R. Instead, we need to express R in the (y, z, x) convention
## To rotate v_prime, we can send it to (x, y, z), rotate it and then send it back to (y, z, x).
## Which means v_prime = U^T R U v. So the rotation matrix in (y, z, x) is R_prime = U^T R U. This is what we can feed to
## e3nn to recover the correct Wigner matrix.
all_wigner = compute_wigner_D(l_max, R, device)
wigner_rotated = apply_wigner_D(all_wigner, spherical_harmonics, l_max=l_max)

#rotated_coords = torch.einsum("b q r, l r-> b l q", R, coordinates[:, [1, 2, 0]])
rotated_coords = torch.einsum("b q r, l r-> b l q", R, coordinates)
matrix_rotated = get_real_spherical_harmonics_e3nn(rotated_coords[0, :, :], l_max)

print(wigner_rotated)
print("\n\n")
print(matrix_rotated)
result_sphericart = get_real_spherical_harmonics(rotated_coords, sh, device, l_max)
print("\n\n")
print(result_sphericart)
"""

"""
sh_values_new = torch.as_tensor(sh.compute(spherical_har_wigner_coord.detach().cpu().numpy()), dtype=torch.float32, device=device)
start_computing = time()
all_wigner = compute_wigner_D(l_max, alpha, beta, gamma)
end_computing = time()
print("Computing time", end_computing - start_computing)
start_new = time()
result = apply_wigner_D(all_wigner, sh_values_new, l_max)
end_new = time()
print("New version", end_new - start_new)
del result
start_new = time()
result = apply_wigner_D(all_wigner, sh_values_new, l_max)
end_new = time()
print("New version", end_new - start_new)



#sh = sct.SphericalHarmonics(l_max=1, normalized=True)
#lat = np.random.uniform(0, np.pi)
#long = np.random.uniform(0, 2*np.pi)
#r = np.random.uniform(0, 10)
#cartesian_coord = convert_spher_cartesian(lat, long, r)
#spherHarmTorch = get_spherical_harmonics(cartesian_coord, sh)

#spherHarmPython = sph_harm(np.array([1], dtype=int), np.array([1], dtype=int), long, lat)
#spherHarmPython_bis = sph_harm(np.array([1], dtype=int), np.array([1], dtype=int), long, lat)

#print(spherHarmTorch)
#print(spherHarmPython)
#print("\n\n\n\n")
#print(-np.sqrt(2)*np.real(spherHarmPython_bis[0]))

grid = torch.randn(( 1, 1, 3), dtype = torch.float32, device=device)
rot_mat = pytorch3d.transforms.random_rotations(1, dtype=torch.float32, device=device)
rotated_grid = torch.einsum("b a q, b r q -> b r a", rot_mat, grid)
result1 = get_real_spherical_harmonics(rotated_grid, sh, device, l_max)

#It works but I have to interchanged the columns to get (Y, Z, X) !!!!!
result_e3nn = e3nn.o3.spherical_harmonics([i for i in range(l_max+1)], rotated_grid[:, :, [1, 2, 0]], normalize=True)

euler_angles = pytorch3d.transforms.matrix_to_euler_angles(rot_mat, "YXY")
euler_angles = euler_angles.detach().cpu()
wigner_matrices = compute_wigner_D(l_max, euler_angles[:,2], euler_angles[:,1], euler_angles[:,0])
#spher = get_real_spherical_harmonics(grid, sh, device, l_max)
spher = e3nn.o3.spherical_harmonics([i for i in range(l_max+1)], grid, normalize=True)
result2 = apply_wigner_D(wigner_matrices, spher[0], l_max)


#print("\n\n\n\n")
#print("One", result1)
#print("E3NN", result_e3nn)
#print("Two", result2)
#print("\n\n")
#print(torch.abs(result1 - result2))


result_e3nn = e3nn.o3.spherical_harmonics(l_max, grid[:, :, [1, 2, 0]], normalize=True)
wign = e3nn.o3.wigner_D(l_max, euler_angles[:,0], euler_angles[:,1], euler_angles[:,2])

print(result_e3nn.shape)
print(wign.shape)
print("\n\n\n")
print(torch.einsum("b q a, b a -> b q", wign.to(device), result_e3nn[:, 0]))
print(e3nn.o3.spherical_harmonics(l_max, rotated_grid[:, :, [1, 2, 0]], normalize=True))

print(euler_angles)
print(e3nn.o3.matrix_to_angles(rot_mat))
"""





