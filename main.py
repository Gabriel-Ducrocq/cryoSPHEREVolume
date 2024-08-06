import torch
import model
import numpy as np
from time import time
from model.grid import Grid, rotate_grid
from model import utils
from model import loss
import wandb
import argparse
import model.utils
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True)
parser_arg.add_argument('--debug', type=bool, required=False)


def train(yaml_setting_path, debug_mode):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, optimizer, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device, \
        scheduler, freqs, freqs_volume, l_max, spherical_harmonics = model.utils.parse_yaml(
        yaml_setting_path)
    if experiment_settings["resume_training"]["model"] != "None":
        name = f"experiment_{experiment_settings['name']}_resume"
    else:
        name = f"experiment_{experiment_settings['name']}"
    if not debug_mode:
        wandb.init(
            # Set the project where this run will be logged
            project="vaeSphericalHarmonics",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=name,

            # Track hyperparameters and run metadata
            config={
                "learning_rate": experiment_settings["optimizer"]["learning_rate"],
                "architecture": "VAE",
                "dataset": experiment_settings["star_file"],
                "epochs": experiment_settings["N_epochs"],
            })
    ############ MODIFYING THINGS TO OVERFIT ONE IMAGES ONLY !!! ###########
    data_loader_std = iter(DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        break

    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd": [], "kl_prior_latent": []}
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DROP LAST !!!!!! ##################################
        data_loader = tqdm(
            iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)))
        start_tot = time()
        for batch_num, (indexes, original_images, batch_images, batch_poses, _) in enumerate(data_loader):
            start_batch = time()
            # start = time()
            ## WHAT I AM DOING HERE IS WRONG, IT IS JUST FOR DEBUGGING
            original_images = original_images.to(device)
            batch_images = batch_images.to(device)
            non_standardized = batch_images.flatten(start_dim=1, end_dim=2)
            batch_images = (batch_images - images_mean)/(images_std + 1e-15)
            batch_poses = batch_poses.to(device)
            flattened_batch_images = batch_images.flatten(start_dim=1, end_dim=2)
            latent_variables, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
            ######### I FIX THE LATENT VARIABLE TO ZERO SINCE THE DATASET IS HOMOGENEOUS !!!!! ###############
            latent_variables = torch.zeros_like(latent_variables)
            alms_per_radius = vae.decode(latent_variables)
            #alms_per_radius = vae.decode(unique_radiuses[None, :, None].repeat(batch_size, 1, 1))
            alms_per_coordinate = utils.alm_from_radius_to_coordinate(alms_per_radius, radius_indexes)
            all_wigner = utils.compute_wigner_D(l_max, batch_poses, device)
            rotated_spherical_harmonics = utils.apply_wigner_D(all_wigner, spherical_harmonics, l_max)
            print("FREQS", freqs.shape)
            all_coordinates = model.grid.rotate_grid(batch_poses, freqs)
            print("rot ", all_coordinates.shape)
            e3nn_rotated_spherical_harmonics = utils.get_real_spherical_harmonics_e3nn(all_coordinates[0], l_max)
            sphericart_coord = sphericartObj.compute(all_coordinates[0].detach().cpu().numpy())
            print(e3nn_rotated_spherical_harmonics[2].shape)
            print(all_wigner[2].shape)
            e3nn_rotated_spherical_harmonics = torch.cat(e3nn_rotated_spherical_harmonics, dim=-1)
            print("Comparison of the two methods")
            print(e3nn_rotated_spherical_harmonics.shape)
            print("Wigner first", e3nn_rotated_spherical_harmonics[1])
            print("Wigner", rotated_spherical_harmonics[0][0])
            print("Rotated by hand", e3nn_rotated_spherical_harmonics[0])
            print("sphericart", sphericart_coord)
            exit()
            all_sph = utils.get_real_spherical_harmonics(all_coordinates, sphericartObj, device, l_max)
            predicted_images = utils.spherical_synthesis_hartley(alms_per_coordinate, all_sph, radius_indexes)
            nll = loss.compute_loss(predicted_images, flattened_batch_images, latent_mean, latent_std, experiment_settings,
                                tracking_metrics, experiment_settings["loss_weights"])
            print("NLL", nll)
            start_grad = time()
            nll.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_grad = time()
            end_batch = time()
            print("Time total:", end_batch - start_batch)
            print("Gradient time:", end_grad - start_grad)

        end_tot = time()
        print("TOTAL TIME", end_tot - start_tot)

        if scheduler:
            scheduler.step()

        if not debug_mode:
            model.utils.monitor_training(tracking_metrics, epoch, experiment_settings, vae, optimizer, device=device,
                    true_images=non_standardized, predicted_images=predicted_images, real_image=original_images,
                                         images_mean=images_mean, images_std=images_std)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    debug_mode = args.debug
    from torch import autograd

    with autograd.detect_anomaly():
        train(path, debug_mode)

