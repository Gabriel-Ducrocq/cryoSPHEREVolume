import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
#import mrc
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



def analyze(yaml_setting_path, model_path, volumes_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, optimizer, dataset, N_epochs, batch_size, sphericartObj, unique_radiuses, radius_indexes, experiment_settings, device, \
        scheduler, freqs, freqs_volume,  l_max = utils.parse_yaml(
        yaml_setting_path)
    vae = torch.load(model_path)
    vae.eval()
    all_coordinates = freqs_volume
    radiuses = torch.sqrt(torch.sum(all_coordinates**2, dim=-1))
    alms_per_coordinate = vae.decode(radiuses[None, :, None])
    all_sph = utils.get_real_spherical_harmonics(all_coordinates[None, :, :], sphericartObj, device, l_max)
    ## I FEED THE RADIUSES DIRECTLY !
    predicted_volume_flattened = utils.spherical_synthesis_hartley(alms_per_coordinate, all_sph, radiuses)
    predicted_volume = predicted_volume_flattened.reshape(190, 190, 190)
    print("VOLUME SHAPE", predicted_volume.shape)

if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    args = parser_arg.parse_args()
    model_path = args.model
    path = args.experiment_yaml
    analyze(path, model_path, None)






