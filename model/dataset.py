import os
import torch
import mrcfile
import numpy as np
from time import time
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from pytorch3d.transforms import euler_angles_to_matrix


class ImageDataSet(Dataset):
    def __init__(self, apix, side_shape, particles_df, particles_path, latent_variables_path, predicted_particles_path = None, down_side_shape=None, down_method="interp", invert_data=True):
        """
        #Create a dataset of images and poses
        #:param apix: float, size of a pixel in Å.
        #:param side_shape: integer, number of pixels on each side of a picture. So the picture is a side_shape x side_shape array
        #:param particle_df: particles dataframe coming from a star file
        #:particles_path: string, path to the folder containing the mrcs files. It is appended to the path present in the star file.
        #:latent_variables_path: str, path to the latent variables. The latent variables file must be ordered in the same way as the starfile.
        #:param down_side_shape: integer, number of pixels of the downsampled images. If no downampling, set down_side_shape = side_shape.
        """

        self.side_shape = side_shape
        self.down_method = down_method
        self.apix = apix
        self.particles_path = particles_path
        self.particles_df = particles_df
        self.predicted_particles_path = predicted_particles_path
        self.latent_variables = torch.tensor(np.load(latent_variables_path), dtype=torch.float32)
        assert self.latent_variables.shape[0] == self.particles_df.shape[0], f"{self.latent_variables.shape[0]} latent variables for {self.particles_df.shape[0]} images."
        print(particles_df.columns)
        #Reading the euler angles and turning them into rotation matrices.
        euler_angles_degrees = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
        euler_angles_radians = euler_angles_degrees*np.pi/180
        poses = euler_angles_to_matrix(torch.tensor(euler_angles_radians, dtype=torch.float32), convention="ZYZ")

        #Reading the translations. ReLion may express the translations divided by apix. So we need to multiply by apix to recover them in Å
        if "rlnOriginXAngst" in particles_df:
            shiftX = torch.as_tensor(np.array(particles_df["rlnOriginXAngst"], dtype=np.float32))
            shiftY = torch.as_tensor(np.array(particles_df["rlnOriginYAngst"], dtype=np.float32))
        else:
            shiftX = torch.as_tensor(np.array(particles_df["rlnOriginX"] * self.apix, dtype=np.float32))
            shiftY = torch.as_tensor(np.array(particles_df["rlnOriginY"] * self.apix, dtype=np.float32))

        self.poses_translation = torch.tensor(torch.vstack([shiftY, shiftX]).T, dtype=torch.float32)
        self.poses = poses
        assert self.poses_translation.shape[0] == self.poses.shape[0], "Rotation and translation pose shapes are not matching !"
        #assert torch.max(torch.abs(poses_translation)) == 0, "Only 0 translation supported as poses"
        print("Dataset size:", self.particles_df.shape[0], "apix:",self.apix)
        print("Normalizing training data")

        #If a downsampling is wanted, recompute the new apix and set the new down_side_shape
        self.down_side_shape = side_shape
        if down_side_shape is not None:
            self.down_side_shape = down_side_shape
            self.down_apix = self.side_shape * self.apix /self.down_side_shape

        self.invert_data = invert_data


    def estimate_image_std(self):
        ### HARD CODED HERE !!!!
        _, hartley_proj, _, _ = self.__getitem__([i for i in range(1000)])
        return torch.std(hartley_proj, dim=0, keepdim=True)


    def __len__(self):
        return self.particles_df.shape[0]

    def __getitem__(self, idx):
        """
        #Return a batch of true images, as 2d array !
        # return: the set of indexes queried for the batch, the corresponding images as a torch.tensor((batch_size, side_shape, side_shape)),
        # the corresponding poses rotation matrices as torch.tensor((batch_size, 3, 3)), the corresponding poses translations as torch.tensor((batch_size, 2))
        # NOTA BENE: the convention for the rotation matrix is left multiplication of the coordinates of the atoms of the protein !!
        """
        particles = self.particles_df.iloc[idx]
        try:
            mrc_idx, img_name = particles["rlnImageName"].split("@")
            mrc_idx = int(mrc_idx) - 1
            mrc_path = os.path.join(self.particles_path, img_name)
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                if mrc.data.ndim > 2:
                    proj = torch.from_numpy(np.array(mrc.data[mrc_idx])).float() #* self.cfg.scale_images
                else:
                    # the mrcs file can contain only one particle
                    proj = torch.from_numpy(np.array(mrc.data)).float() #* self.cfg.scale_images

            # get (1, side_shape, side_shape) proj
            if len(proj.shape) == 2:
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            else:
                assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

            if self.down_side_shape != self.side_shape:
                if self.down_method == "interp":
                    proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)
                #elif self.down_method == "fft":
                #    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
                else:
                    raise NotImplementedError

            proj = proj[0]
            r = torch.fft.ifftshift(proj, dim=(-2, -1))
            fourier_proj = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])),
                                                dim=(-2, -1))
            hartley_proj = fourier_proj.real - fourier_proj.imag




        except Exception as e:
            print(f"WARNING: Particle image {img_name} invalid! Setting to zeros.")
            print(e)
            hartley_proj = torch.zeros(self.down_side_shape, self.down_side_shape)

        #if self.invert_data:
        #    print("INVERTING")
        #    proj *= -1

        #       !!!!!!!!!!!!!        FOR NOW, THE PREDICTED PARTICLES NEED TO BE ORDERED IN THE SAME WAY AS THE STAR FILE    !!!!!!
        #try:
        with mrcfile.mmap(self.predicted_particles_path, mode="r", permissive=True) as mrc:
            if mrc.data.ndim > 2:
                predicted_proj = torch.from_numpy(np.array(mrc.data[idx])).float() #* self.cfg.scale_images
                print("PREDICTED PROJ", predicted_proj.shape)
            else:
                # the mrcs file can contain only one particle
                predicted_proj = torch.from_numpy(np.array(mrc.data)).float() #* self.cfg.scale_images

        # get (1, side_shape, side_shape) proj
        if len(predicted_proj.shape) == 2:
            predicted_proj = predicted_proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
        else:
            assert len(predicted_proj.shape) == 3 and predicted_proj.shape[0] == 1  # some starfile already have a dummy channel

        if self.down_side_shape != self.side_shape:
            if self.down_method == "interp":
                predicted_proj = tvf.resize(predicted_proj, [self.down_side_shape, ] * 2, antialias=True)
                #elif self.down_method == "fft":
                #    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
            else:
                raise NotImplementedError

        #except Exception as e:
        #    print(f"WARNING: Particle image {self.predicted_particles_path} invalid! Setting to zeros.")
        #    print(e)
        #    predicted_proj = torch.zeros(self.down_side_shape, self.down_side_shape)

        return idx, proj, hartley_proj, self.poses[idx], self.poses_translation[idx]/self.down_apix, self.latent_variables[idx], predicted_proj
