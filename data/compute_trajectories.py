import mrcfile
import numpy as np
import yaml
import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from analyze import decode
path = os.path.abspath("model")
sys.path.append(path)
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_mean_std(experiment_yaml):
    (vae, optimizer, image_translator, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device,
    scheduler, freqs, freqs_volume, l_max, spherical_harmonics, wigner_calculator, ctf, use_ctf, circular_mask) = utils.parse_yaml(experiment_yaml)

    data_loader_std = iter(DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4, drop_last=True))
    for batch_num, (indexes, original_images, images_for_std, batch_poses, _) in enumerate(data_loader_std):
        images_std = torch.std(images_for_std).to(device)
        images_mean = torch.mean(images_for_std).to(device)
        break

    return images_mean, images_std

def load_model(model_path):
    return torch.load(model_path)

def read_images(images_path, list_interest):
    print(list_interest)
    with mrcfile.open(images_path) as mrc:
        images_interest = mrc.data[list_interest]

    return torch.tensor(images_interest, dtype=torch.float32, device=device)

def sample_latent(batch_images, model, images_mean, images_std):
    batch_images = batch_images.to(device)
    batch_images = (batch_images - images_mean)/(images_std + 1e-15)
    flattened_batch_images = batch_images.flatten(start_dim=1, end_dim=2)
    latent_variables, latent_mean, latent_std = model.sample_latent(flattened_batch_images)
    return latent_means

def compute_trajectories(experiment_yaml_path, models_path, images_list, epochs_list, output_path):
    images_mean, images_std = compute_mean_std(experiment_yaml_path)
    with open(experiment_yaml_path, "r") as file:
        experiment_settings = yaml.safe_load(file)

    batch_images = read_images(experiment_settings["particles_path"] + "particles.mrcs", images_list)
    for epoch in epochs_list:
        output_epoch = output_path + f"epoch_{epoch}"
        if not os.path.exists(output_epoch):
            os.makedirs(output_epoch)

        print("Epoch:", epoch)
        model = load_model(models_path + f"full_model{epoch}")
        model.eval()
        latent_means = sample_latent(batch_images, model, images_mean, images_std)
        for i, latent_mean in enumerate(latent_means):
            decode(experiment_yaml, latent_mean[None, :], model_path, output_path=output_epoch + f"volume_{images_list[i]}.mrc")





if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--models_path", type=str, required=True)
    parser_arg.add_argument("--epochs_list", nargs='+', required=True, type=int)
    parser_arg.add_argument("--images_list", nargs='+', required=True, type=int)
    parser_arg.add_argument("--output_path", type=str, required=True)
    args = parser_arg.parse_args()
    models_path = args.models_path
    experiment_yaml = args.experiment_yaml
    epochs_list = args.epochs_list
    images_list = args.images_list
    output_path = args.output_path
    compute_trajectories(experiment_yaml, models_path, images_list, epochs_list, output_path)