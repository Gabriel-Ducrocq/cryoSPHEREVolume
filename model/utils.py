import torch
import sphericart as sct

import wandb
import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
from dataset import ImageDataSet
from vae import VAE
from mlp import MLP
import yaml
from grid import Grid
import numpy as np
#from astropy.coordinates import cartesian_to_spherical
import starfile

def get_radius_indexes(freqs, device):
    """
    Link the index of the unique indexes to the corresponding frequencies
    :param freqs: torch.tensor(side_hape**2, 3)
    :return: torch.tensor(side_shape**2)
    """
    radius = torch.sqrt(torch.sum(freqs ** 2, axis=-1))
    unique_radius = torch.unique(radius, sorted=True)
    unique_indexes = torch.linspace(0, len(unique_radius)-1, len(unique_radius), dtype=torch.int, device=device)
    rad_and_ind = torch.stack([unique_radius, unique_indexes], dim=-1)
    indexes = torch.stack([rad_and_ind[rad_and_ind[:, 0] == rad, 1] for rad in radius], dim=0)
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

    l_max = experiment_settings["l_max"]
    particles_path = experiment_settings["particles_path"]
    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    apix_downsize = Npix * apix /Npix_downsize

    frequencies = Grid(image_settings["Npix_downsize"], image_settings["apix"], device)
    radius_indexes, unique_radiuses = get_radius_indexes(frequencies.freqs, device)
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



    return vae, optimizer, dataset, N_epochs, batch_size, sh, radius_indexes, experiment_settings, device, \
    scheduler, frequencies.freqs, l_max

def get_real_spherical_harmonics(coordinates, sphericart_obj, device, l_max):
    """
    Computes the real, cartesian spherical harmonics functions.
    :param coordinates: torch.tensor(N_batch, N_freqs, 3) where N_freqs can be N_side_freq**3 if volume reconstruction
                        and N_side_freq**2 if image reconstruction
    :param sphericart_obj: sphericat object for spherical harmonics computation, until a defined l_max, normalized or not
    :return: torch.tensor(N_batch, N_freqs)
    """
    batch_size = coordinates.shape[0]
    coordinates = coordinates.reshape(-1, 3)
    sh_values = torch.as_tensor(sphericart_obj.compute(coordinates.detach().cpu().numpy()), dtype=torch.float32, device=device).reshape(batch_size, -1, (l_max+1)**2)
    return sh_values


def alm_from_radius_to_coordinate(alm, radiuses_index):
    """
    The alm that the network outputs are on per radius. We need to match the coordinate to the radiuses
    :param alm: torch.tensor(N_batch, N_unique_radius, (l_max+1)**2)
    :param radiuses_index: torch.tensor(side_shape**2) of alm index corresponding to the radius of that coordinate
    :return: torch.tensor(N_batch, side_shape**2, (l_max+1)**2)
    """
    return alm[:, radiuses_index, :]

def spherical_synthesis_hartley(alm_per_coord, spherical_harmonics, indexes):
    """
    Computes the Hartley transform through a spherical harmonics synthesis
    :param alm: torch.tensor(N_batch, side_shape**2, (l_max+1)**2)
    :param spherical_harmonics:torch.tensor(N_batch, side_shape**2, (l_max+1)**2)
    :param radiuses_index: torch.tensor(side_shape**2) of alm index corresponding to the radius of that coordinate
    :return: torch.tensor(N_batch, side_shape**2)
    """
    #Here, the frequencies (0,0) gave NaN for the (l_max+1)**2 coefficients, except l = 0. We replace directly with the
    #estimates provided by the neural net
    spherical_harmonics[:, indexes ==0, :] = 0
    print("SH MAX", torch.max(spherical_harmonics))
    print("SH MIN", torch.min(spherical_harmonics))
    print("ALMS", alm_per_coord)
    print("ALMS MIN", torch.min(alm_per_coord))
    print("ALMS MIN", torch.max(alm_per_coord))
    images_radius_0_nan = torch.einsum("b s l, b s l -> b s", alm_per_coord, spherical_harmonics)
    images_radius_0_nan[:, indexes == 0] = alm_per_coord[:, indexes == 0, 0]
    return images_radius_0_nan


def hartley_to_fourier(image, device):
    """
    Converts a Hartley image into a FFT image
    :param image: torch.tensor(batch_size, side_shape**2). Once reshaped to (side_shape, side_shape), the frequency 0 is at the center
    :return: torch.tensor(batch_size, side_shape, side_shape)
    """
    side_shape = int(np.sqrt(image.shape[1]))
    batch_size = image.shape[0]
    assert side_shape % 2 == 0, "Image must have an even number of pixels"
    ## The image now has zero at the center and the lowest frequency at the top left.
    image_square = image.reshape(batch_size, side_shape, side_shape)
    low_x = image_square[:, 0, :]
    low_y = image_square[:, :, 0]
    ## Since the Fourier transform is side_shape periodic, we can safely remove the lowest frequency and deal with it
    ## separately
    image_cropped = image_square[:, 1:, 1:]
    image_cropped_flipped = image_cropped.flip(dims=(1, 2))
    fourier_transform = torch.view_as_complex(torch.zeros_like(image_square, dtype=torch.float32, device = device))
    fourier_transform[:, 1:, 1:] = (image_cropped + image_cropped_flipped)/2 - 1j*(image_cropped - image_cropped_flipped)/2
    fourier_transform[:, 0, :] = (low_x + low_y)/2 - 1j*(low_x-low_y)/2
    fourier_transform[:, :, 0] = (low_y + low_x)/2 -1j*(low_y - low_x)/2
    return fourier_transform

def fourier_to_real(fft_images):
    """
    Compute the inverse fft to recover the image
    :param fft_images: torch.tensor(batch_size, side_shape, side_shape) of images in fourier space
    :return: torch.tensor(batch_size, side_shape, side_shape)
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(fft_images, dim=(-1, -2)))).real

def hartley_to_real(images, device):
    """
    Goes from Hartley space to real space
    :param images: torch.tensor(batch_size, side_shape, side_shape) of images in Hartley space.
    :param device: device, either gpu or cpu
    :return: torch.tensor(batch_size, side_shape, side_shape) of images in real space.
    """
    fft_images = hartley_to_fourier(images, device)
    return fourier_to_real(fft_images)

def monitor_training(tracking_metrics, epoch, experiment_settings, vae, optimizer, device=None, true_images=None, predicted_images=None, real_image=None):
    """
    Monitors the training process through wandb and saving masks and models
    :param mask:
    :param tracking_metrics:
    :param epoch:
    :param experiment_settings:
    :param vae:
    :return:
    """
    real_image_again = hartley_to_real(true_images[:1], device)
    real_predicted_image = hartley_to_real(predicted_images[:1], device)
    wandb.log({key: np.mean(val) for key, val in tracking_metrics.items()})
    wandb.log({"epoch": epoch})
    wandb.log({"lr":optimizer.param_groups[0]['lr']})
    print("TESTESTESTEST")
    print(real_image_again[0].detach().cpu().numpy()[:, :, None].shape)
    real_image_again_wandb = wandb.Image(real_image_again[0].detach().cpu().numpy()[:, :, None], caption="Round trip real to Hartley")
    true_image_ony_real_wandb = wandb.Image(real_image[0].detach().cpu().numpy()[:, :, None],
                                         caption="Original image")
    predicted_image_wandb = wandb.Image(real_predicted_image[0].detach().cpu().numpy()[:, :, None],
                                         caption="Predicted images")
    wandb.log({"Images/true_image": real_image_again_wandb})
    wandb.log({"Images/true_image_ony_real": true_image_ony_real_wandb})
    wandb.log({"Images/predicted_image": predicted_image_wandb})
    torch.save(vae, experiment_settings["folder_path"] + "models/full_model" + str(epoch))



def convert_spher_cartesian(lat, long, r):
    return np.array((r*np.sin(lat)*np.cos(long), r*np.sin(lat)*np.sin(long), r*np.cos(lat)))[None, :]



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




