import torch


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim = None, lmax=9):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.latent_dim = latent_dim
        self.lmax = lmax
        self.elu = torch.nn.ELU()

    def sample_latent(self, images):
        """
        Samples latent variables given an image
        :param images: torch.tensor(N_batch, side_shape**2)
        :return: torch.tensor(N_batch, latent_dim) latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std if latent_type is "continuous"
        """
        latent_mean, latent_std = self.encoder(images)
        latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                            *latent_std

        return latent_variables, latent_mean, latent_std


    def decode(self, latent_variables):
        """
        Decode the latent variables
        :param latent_variables: torch.tensor(N_batch, n_coords, latent_dim + 3*positional_embedding_dim) or torch.tensor(N_batch, n_coords, latent_dim + 3*positional_embedding_dim+ 3)
                                if we keep the original coordinates with the positional embedding.
        :return: torch.tensor(N_batch, n_coords, 1) of decoder evaluated in the coordinates
        """
        return self.decoder(latent_variables)




