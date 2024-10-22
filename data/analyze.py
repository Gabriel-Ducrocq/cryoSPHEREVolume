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


def decode(yaml_setting_path, all_latent_variables, model_path):
    vae, optimizer, image_translator, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device, \
    scheduler, freqs, freqs_volume, l_max, spherical_harmonics, wigner_calculator, ctf, use_ctf, circular_mask = model.utils.parse_yaml(yaml_setting_path)

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


    all_latent_variables = torch.tensor(all_latent_variables, dtype=torch.float32, device=device)
    vae = torch.load(model_path)
    vae.eval()

    all_coordinates = freqs_volume
    all_radiuses_volumes = torch.sqrt(torch.sum(all_coordinates**2, dim=1))

    for k, latent_variables in enumerate(all_latent_variables):
        latent_variables = latent_variables[None, :]
        alms_per_radius = vae.decode(latent_variables)
        #The next tensor is ((l_max+1)**2, N_coordinates)
        ## I transposed the alm per radius: it is of shape (N_unique_radiuses, (l_max+1)**2)
        alms_radiuses_volume = []
        for l in range((l_max+1)**2):
            linearInterpolator = interp.Interp1D(unique_radiuses, alms_per_radius[0, :, l],
                                                 method="linear", extrap=0.0)
            #We only interpolate the frequencies within the circular mask !
            alms_radiuses_volume_l = linearInterpolator(all_radiuses_volumes[circular_mask.mask_volume ==1])
            alms_radiuses_volume.append(alms_radiuses_volume_l)

        del alms_per_radius
        alms_radiuses_volume = torch.stack(alms_radiuses_volume, dim=1)[None, :, :]
        print("COORDINSTES", all_coordinates.shape)
        torch.cuda.empty_cache()
        all_chunks_sph = []
        predicted_volume_hartley_flattened = []
        total_number_freqs = torch.sum(circular_mask.mask_volume)
        all_target_coordinates = all_coordinates[circular_mask.mask_volume]
        n_iterations = total_number_freqs // 1000
        assert n_iterations > 0, "There is not enough frequencies."
        for i in range(1001):
            start = i*total_number_freqs
            end = i*total_number_freqs+ total_number_freqs
            print("all_coordinates shape", all_target_coordinates[start:end].shape)
            all_sph = utils.get_real_spherical_harmonics(all_target_coordinates[start:end], sphericartObj, device, l_max)
            all_sph = torch.cat(all_sph, dim=-1)
            print("SHAPESSSSSS AGAIN")
            print(all_sph.shape)
            print(alms_radiuses_volume.shape)
            predicted_volume_hartley_flattened_slice = torch.einsum("b s l, s l -> b s", alms_radiuses_volume[:, start:end, :], all_sph)
            #all_chunks_sph.append(all_sph)
            predicted_volume_hartley_flattened.append(predicted_volume_hartley_flattened_slice)

        del all_coordinates
        del predicted_volume_hartley_flattened_slice
        print("SIZE OF HARTLEY", predicted_volume_hartley_flattened[0].shape)
        #all_sph = torch.cat(all_chunks_sph, dim=0)
        predicted_volume_hartley_flattened = torch.cat(predicted_volume_hartley_flattened, dim=1)
        del all_chunks_sph
        ## I FEED THE RADIUSES DIRECTLY !
        #predicted_volume_hartley_flattened = torch.einsum("b s l, s l -> b s", alms_radiuses_volume, all_sph)
        predicted_volume_hartley_flattened[:, all_radiuses_volumes == 0.0] = 0
        predicted_volume_hartley_flattened *= images_std
        predicted_volume_hartley_flattened += images_mean
        all_freqs_volume_hartley_flattened = torch.zeros_like(190*190*190)
        all_freqs_volume_hartley_flattened[circular_mask.mask_volume == 1 ] = predicted_volume_hartley_flattened
        predicted_volume_hartley = predicted_volume_hartley_flattened.reshape(190, 190, 190)
        print("Hartley shape", predicted_volume_hartley.shape)
        predicted_volume_real = utils.hartley_transform_3d(predicted_volume_hartley[None, :, :])
        print("VOLUME SHAPE", predicted_volume_real.shape)
        folder_experiment = "data/dataset/"
        mrc.MRCFile.write(f"{folder_experiment}volume_{k}.mrc", predicted_volume_real[0].detach().cpu().numpy(), Apix=1.0, is_vol=True)
        del predicted_volume_real


def compute_latent_variables(yaml_setting_path, model_path):
    vae, optimizer, image_translator, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device, \
    scheduler, freqs, freqs_volume, l_max, spherical_harmonics, wigner_calculator, ctf, use_ctf, circular_mask = model.utils.parse_yaml(yaml_setting_path)

    data_loader_std = iter(DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        break

    vae = torch.load(model_path)
    vae.eval()
    all_latent_variables = []
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False))
    for batch_num, (indexes, original_images, batch_images, batch_poses, _) in enumerate(data_loader):
        batch_images = batch_images.to(device)
        batch_images = (batch_images - images_mean)/(images_std + 1e-15)
        flattened_batch_images = batch_images.flatten(start_dim=1, end_dim=2)
        latent_variables, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
        all_latent_variables.append(latent_mean)

    all_latent_variables = torch.concat(all_latent_variables, dim=0)
    np.save("data/dataset/z.npy", all_latent_variables.detach().cpu().numpy())





def analyze(yaml_setting_path, model_path, encode, latent_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """

    if encode:
        compute_latent_variables(yaml_setting_path, model_path)
    else:
        latent_variables = np.load(latent_path)
        decode(yaml_setting_path, latent_variables, model_path)

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
    parser_arg.add_argument("--latent_path", type=str, required=False)
    parser_arg.add_argument('--encode', action=argparse.BooleanOptionalAction)
    args = parser_arg.parse_args()
    model_path = args.model
    path = args.experiment_yaml
    encode = args.encode
    latent_path = args.latent_path
    analyze(path, model_path, encode, latent_path)






