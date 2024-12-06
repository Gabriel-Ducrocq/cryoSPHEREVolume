import torch
import model
import numpy as np
from time import time
from model.grid import Grid, rotate_grid
from model import utils
from model import loss
from model import renderer
import wandb
import argparse
import model.utils
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True)
parser_arg.add_argument('--debug', type=bool, required=False)


def train(yaml_setting_path, debug_mode):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    decoder, optimizer, image_translator, dataset, N_epochs, batch_size, unique_radiuses, radius_indexes, experiment_settings, device, \
        scheduler, freqs, freqs_volume, ctf, use_ctf, circular_mask, grid, pos_encoding, mask_radius = model.utils.parse_yaml(
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
    ####### I LOOK AT THE FIRST 6000 images !!!!! #######
    data_loader_std = iter(DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _, batch_latent_variables, batch_structural_predicted_images) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        structural_images_std = torch.std(batch_structural_predicted_images).to(device)
        structural_images_mean = torch.mean(batch_structural_predicted_images).to(device)
        break

    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd": [], "kl_prior_latent": [], "rmsd_structural":[]}
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DROP LAST !!!!!! ##################################
        data_loader = tqdm(
            iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)))

        start_tot = time()
        for batch_num, (indexes, original_images, batch_images, batch_poses, batch_poses_translation, batch_latent_variables, batch_structural_predicted_images) in enumerate(data_loader):
            start_batch = time()
            original_images = original_images.to(device)
            batch_images = batch_images.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            batch_latent_variables = batch_latent_variables.to(device)
            non_standardized = batch_images.flatten(start_dim=1, end_dim=2)
            batch_images = (batch_images - images_mean)/(images_std + 1e-15)
            batch_poses = batch_poses.to(device)
            flattened_batch_images = batch_images.flatten(start_dim=1, end_dim=2)
            images_real = model.utils.real_to_hartley(batch_images)
            batch_translated_images_real = image_translator.transform(images_real, batch_poses_translation[:, None, :])
            batch_translated_images_hartley = model.utils.real_to_hartley(batch_translated_images_real)
            batch_translated_images_hartley = (batch_translated_images_hartley - images_mean)/(images_std + 1e-15)
            batch_translated_images_hartley = batch_translated_images_hartley.flatten(start_dim=1, end_dim=2)
            print("BATCH STRUCTURAL IMAGE", batch_structural_predicted_images.shape)
            batch_structural_predicted_images = (batch_structural_predicted_images.to(device) - structural_images_mean)/(structural_images_std + 1e-15)

            mask = circular_mask.get_mask(mask_radius)
            rotated_grid = rotate_grid(batch_poses, grid.freqs[mask==1].to(device))
            coordinates_embedding = pos_encoding(rotated_grid)

            decoder_input = torch.cat([coordinates_embedding, batch_latent_variables[:, None, :].expand(-1, rotated_grid.shape[1], -1)], dim=-1)
            decoded_images = decoder(decoder_input)
            predicted_images = torch.zeros((batch_size, batch_images.shape[1]*batch_images.shape[2]), dtype=torch.float32, device=device)
            predicted_images[:, mask==1] = decoded_images[:, :, 0]
            predicted_images = predicted_images.reshape(batch_images.shape)
            if use_ctf:
                batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
            else:
                batch_predicted_images = predicted_images


            nll = loss.compute_loss(batch_predicted_images.flatten(start_dim=1, end_dim=2), batch_translated_images_hartley, batch_structural_predicted_images, predicted_images, tracking_metrics)
            print("NLL", nll)
            start_grad = time()
            nll.backward()
            optimizer.step()
            optimizer.zero_grad()

        end_tot = time()
        print("TOTAL TIME", end_tot - start_tot)

        if scheduler:
            scheduler.step()

        if not debug_mode:
            model.utils.monitor_training(decoder, tracking_metrics, epoch, experiment_settings, optimizer, device=device,
                    true_images=non_standardized, predicted_images=predicted_images.flatten(start_dim=1, end_dim=2), real_image=original_images,
                                         images_mean=images_mean, images_std=images_std, structural_predicted_images=batch_structural_predicted_images)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    debug_mode = args.debug
    from torch import autograd

    with autograd.detect_anomaly():
        train(path, debug_mode)

