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
from ctf import CTF
from time import time
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix

class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True

def concat_and_save(tens, path):
    """
    Concatenate the lsit of tensor along the dimension 0
    :param tens: list of tensor with batch size as dim 0
    :param path: str, path to save the torch tensor
    :return: tensor of concatenated tensors
    """
    concatenated = torch.concat(tens, dim=0)
    np.save(path, concatenated.detach().numpy())
    return concatenated


filter_aa = True



def analyze(yaml_setting_path, model_path, volumes_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, optimizer, dataset, N_epochs, batch_size, sphericartObj, radius_indexes, experiment_settings, device, \
        scheduler, freqs, l_max = model.utils.parse_yaml(
        yaml_setting_path)
    vae = torch.load(model_path)
    vae.eval()
    all_coordinates = freqs.freqs_volume
    radiuses = torch.sqrt(torch.sum(all_coordinates**2, dim=-1))
    alms_per_coordinate = vae.decode(radiuses[None, :, None])
    all_sph = utils.get_real_spherical_harmonics(all_coordinates, sphericartObj, device, l_max)
    predicted_volume_flattened = utils.spherical_synthesis_hartley(alms_per_coordinate, all_sph, radius_indexes)
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





