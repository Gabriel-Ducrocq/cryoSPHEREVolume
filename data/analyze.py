import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import argparse
import starfile
import numpy as np
from time import time
from tqdm import tqdm
import xitorch.interpolate as interp
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix
import matplotlib.pyplot as plt



def analyze(yaml_setting_path, model_path, volumes_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, optimizer, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device, \
        scheduler, freqs, freqs_volume, l_max, spherical_harmonics, wigner_calculator = utils.parse_yaml(
        yaml_setting_path)

    data_loader_std = iter(DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        break

    del original_images
    del images_for_std
    del batch_poses
    del data_loader_std
    del indexes
    del batch_num
    vae = torch.load(model_path)
    vae.eval()

    all_coordinates = freqs_volume
    all_radiuses_volumes = torch.sqrt(torch.sum(all_coordinates**2, dim=1))
    alms_per_radius = vae.decode(torch.zeros((1, 8), dtype=torch.float32, device=device))
    #The next tensor is ((l_max+1)**2, N_coordinates)
    ## I transposed the alm per radius: it is of shape (N_unique_radiuses, (l_max+1)**2)
    alms_radiuses_volume = []
    for l in range((l_max+1)**2):
        linearInterpolator = interp.Interp1D(unique_radiuses, alms_per_radius[0, :, l],
                                             method="linear", extrap=0.0)
        alms_radiuses_volume_l = linearInterpolator(all_radiuses_volumes)
        print("Interpolation probleö number:", l)
        print(alms_radiuses_volume_l.shape)
        alms_radiuses_volume.append(alms_radiuses_volume_l)

    del alms_per_radius
    del all_radiuses_volumes
    del wigner_calculator
    alms_radiuses_volume = torch.stack(alms_radiuses_volume, dim=1)[None, :, :]
    print("COORDINSTES", all_coordinates.shape)
    torch.cuda.empty_cache()
    all_sph = utils.get_real_spherical_harmonics(all_coordinates, sphericartObj, device, l_max)
    all_sph = torch.cat(all_sph, dim=-1)
    ## I FEED THE RADIUSES DIRECTLY !
    predicted_volume_hartley_flattened = torch.einsum("b s l, s l -> b s", alms_radiuses_volume, all_sph)
    predicted_volume_hartley_flattened[:, all_radiuses_volumes == 0.0] = 0
    predicted_volume_hartley = predicted_volume_hartley_flattened.reshape(190, 190, 190)
    predicted_volume_hartley *= images_std
    predicted_volume_hartley += images_mean
    print("Hartley shape", predicted_volume_hartley.shape)
    predicted_volume_real = utils.hartley_transform_3d(predicted_volume_hartley[None, :, :])

    print("VOLUME SHAPE", predicted_volume_real.shape)
    folder_experiment = "data/dataset/"
    mrc.MRCFile.write(f"{folder_experiment}volume.mrc", predicted_volume_real[0].detach().cpu().numpy(), Apix=1.0, is_vol=True)

    """
    ######### I FIX THE LATENT VARIABLE TO ZERO SINCE THE DATASET IS HOMOGENEOUS !!!!! ###############
    latent_variables = torch.zeros((1, 8), dtype=torch.float32, device=device)
    alms_per_radius = vae.decode(latent_variables)
    # alms_per_radius = vae.decode(unique_radiuses[None, :, None].repeat(batch_size, 1, 1))
    alms_per_coordinate = utils.alm_from_radius_to_coordinate(alms_per_radius, radius_indexes)
    del alms_per_radius
    del latent_variables
    #all_wigner = utils.compute_wigner_D(l_max, batch_poses, device)
    #rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
    spherical_harmonics = torch.cat(spherical_harmonics, dim=-1)[None, :, :]
    predicted_images_flattened = utils.spherical_synthesis_hartley(alms_per_coordinate, spherical_harmonics,
                                                         radius_indexes)
    real_predicted_image = utils.hartley_to_real(predicted_images_flattened, device, images_mean, images_std)
    #### !!!!!! I AM MSSING THE STANDARDIZATION AND THE GOING FROM HARtLEY TO REAL !!!!!!! #######
    plt.imshow(real_predicted_image[0].detach().cpu().numpy(), cmap="gray")
    plt.savefig("data/dataset/image.png")
    plt.show()

    print("SHAPE OF SPHER", spherical_harmonics.shape)
    real_predicted_image = real_predicted_image.repeat(190, 1, 1)
    folder_experiment = "data/dataset/"
    mrc.MRCFile.write(f"{folder_experiment}stacked_image.mrc", real_predicted_image.detach().cpu().numpy(), Apix=1.0, is_vol=True)
    """

if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    args = parser_arg.parse_args()
    model_path = args.model
    path = args.experiment_yaml
    analyze(path, model_path, None)






