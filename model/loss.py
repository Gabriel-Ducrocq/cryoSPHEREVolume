import torch
import utils

def compute_image_loss(true_image, predicted_image):
    """
    Compute the mean squared error loss over the batch between true and predicted images, in Hartley space
    :param true_image: torch.tensor(N_batch, side_shape**2), in Hartley space
    :param predicted_image: torch.tensor(N_batch, side_shape**2), in Hartley space
    :return: torch.tensor(1) of the average over the batch of the squared error loss between images
    """
    return torch.mean(torch.mean((true_image - predicted_image)**2, dim=-1))


def compute_KL_prior_latent(latent_mean, latent_std, epsilon_loss):
    """
    Computes the KL divergence between the approximate posterior and the prior over the latent,
    where the latent prior is given by a standard Gaussian distribution.
    :param latent_mean: torch.tensor(N_batch, latent_dim), mean of the Gaussian approximate posterior
    :param latent_std: torch.tensor(N_batch, latent_dim), std of the Gaussian approximate posterior
    :param epsilon_loss: float, a constant added in the log to avoid log(0) situation.
    :return: torch.float32, average of the KL losses accross batch samples
    """
    return torch.mean(-0.5 * torch.sum(1 + torch.log(latent_std ** 2 + eval(epsilon_loss)) \
                                           - latent_mean ** 2 \
                                           - latent_std ** 2, dim=1))


def compute_loss(predicted_images, images, latent_mean, latent_std, experiment_settings, tracking_dict, loss_weights):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, side_shape**2), predicted images
    :param images: torch.tensor(batch_size, side_shape**2), images
    :param latent_mean:torch.tensor(batch_size, latent_dim), mean of the approximate latent distribution
    :param latent_std:torch.tensor(batch_size, latent_dim), std of the approximate latent distribution
    :param loss_weights: dict, containing the strength of losses for each loss
    :return:
    """
    print("TRUE IMAGES", images)
    print("PREDICTED IMAGES", predicted_images)
    rmsd = compute_image_loss(images, predicted_images)
    KL_prior_latent = compute_KL_prior_latent(latent_mean, latent_std, experiment_settings["epsilon_kl"])

    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_latent"].append(KL_prior_latent.detach().cpu().numpy())

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent/images.shape[1]
    return loss