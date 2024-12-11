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
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix
import matplotlib.pyplot as plt


def decode(yaml_setting_path, all_latent_variables, model_path, output_path):
    decoder, optimizer, image_translator, dataset, N_epochs, batch_size, unique_radiuses, radius_indexes, experiment_settings, device, \
        scheduler, freqs, freqs_volume, ctf, use_ctf, circular_mask, grid, pos_encoding, mask_radius = utils.parse_yaml(
        yaml_setting_path)


    all_latent_variables = torch.tensor(all_latent_variables, dtype=torch.float32, device=device)
    decoder = torch.load(model_path)
    decoder.eval()
    mask = circular_mask.get_mask_3d(mask_radius)
    freqs_eval = grid.freqs_volume[mask==1]
    coordinates_embedding = pos_encoding(freqs_eval)
    for k, z in enumerate(all_latent_variables):
        z = z[None, :, :]
        decoder_input = torch.cat([coordinates_embedding, z[:, None, :].expand(-1, freqs_eval.shape[1], -1)], dim=-1)
        decoded_volume = decoder(decoder_input)
        predicted_volume = torch.zeros((batch_size, grid.side_shape**3), dtype=torch.float32, device=device)
        predicted_volume[:, mask==1] = decoded_volume[:, :, 0]
        predicted_volume = predicted_images.reshape(1, grid.side_shape, grid.side_shape, grid.side_shape)
        mrc.MRCFile.write(f"{output_path}volume_{k}.mrc", predicted_volume_real[0].detach().cpu().numpy(), Apix=1.0, is_vol=True)


def analyze(yaml_setting_path, model_path, latent_path, volume_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    latent_variables = np.load(latent_path)
    decode(yaml_setting_path, latent_variables, model_path, volume_path)

if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    parser_arg.add_argument("--latent_path", type=str, required=False)
    parser_arg.add_argument("--volume_path", type=str, required=False)
    args = parser_arg.parse_args()
    model_path = args.model
    path = args.experiment_yaml
    latent_path = args.latent_path
    volume_path = args.volume_path
    analyze(path, model_path, latent_path, volume_path)






