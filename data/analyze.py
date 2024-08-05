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
        scheduler, freqs, freqs_volume,  l_max = utils.parse_yaml(
        yaml_setting_path)

    data_loader_std = iter(DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        break

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

    alms_radiuses_volume = torch.stack(alms_radiuses_volume, dim=1)[None, :, :]
    print("COORDINSTES", all_coordinates.shape)
    all_sph = utils.get_real_spherical_harmonics(all_coordinates[None, :, :], sphericartObj, device, l_max)
    ## I FEED THE RADIUSES DIRECTLY !
    predicted_volume_hartley_flattened = torch.einsum("b s l, b s l -> b s", alms_radiuses_volume, all_sph)
    predicted_volume_hartley_flattened[:, all_radiuses_volumes == 0.0] = 0
    predicted_volume_hartley = predicted_volume_hartley_flattened.reshape(190, 190, 190)
    predicted_volume_hartley *= images_std
    predicted_volume_hartley += images_mean
    print("Hartley shape", predicted_volume_hartley.shape)
    predicted_volume_fourier = utils.hartley_to_fourier_3d(predicted_volume_hartley, device)
    predicted_volume_real = torch.fft.fftn(predicted_volume_fourier).real

    print("VOLUME SHAPE", predicted_volume_real.shape)
    folder_experiment = "data/dataset/"
    mrc.MRCFile.write(f"{folder_experiment}test.mrc", predicted_volume_real.detach().cpu().numpy(), Apix=1.0, is_vol=True)


    all_coordinates = freqs
    radiuses = torch.sqrt(torch.sum(all_coordinates**2, dim=-1))
    alms_per_coordinate = vae.decode(radiuses[None, :, None])
    print("COORDINSTES", all_coordinates.shape)
    all_sph = utils.get_real_spherical_harmonics(all_coordinates[None, :, :], sphericartObj, device, l_max)
    predicted_image_flattened = utils.spherical_synthesis_hartley(alms_per_coordinate, all_sph, radius_indexes)
    #### !!!!!! I AM MSSING THE STANDARDIZATION AND THE GOING FROM HARtLEY TO REAL !!!!!!! #######
    predicted_image = predicted_image_flattened.reshape(190, 190)
    plt.imshow(predicted_image.detach().cpu().numpy(), cmap="gray")
    plt.savefig("data/dataset/image.png")
    plt.show()


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    args = parser_arg.parse_args()
    model_path = args.model
    path = args.experiment_yaml
    analyze(path, model_path, None)






