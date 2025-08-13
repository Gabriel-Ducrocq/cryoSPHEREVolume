import torch
import utils
import matplotlib.pyplot as plt

def calc_cor_loss(pred_images, gt_images, mask=None):
    """
    Compute the cross-correlation for each pair (predicted_image, true) image in a batch. And average them
    pred_images: torch.tensor(batch_size, side_shape**2) predicted images
    gt_images: torch.tensor(batch_size, side_shape**2) of true images, translated according to the poses.
    return torch.tensor(1) of average correlation accross the batch.
    """
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]

    pred_images = torch.flatten(pred_images, start_dim=-2, end_dim=-1)
    gt_images = torch.flatten(gt_images, start_dim=-2, end_dim=-1)
    # b, h, w -> b, num_pix
    #pred_images = pred_images.flatten(start_dim=2)
    #gt_images = gt_images.flatten(start_dim=2)

    # b 
    dots = (pred_images * gt_images).sum(-1)
    # b -> b 
    err = -dots / (gt_images.std(-1) + 1e-5) / (pred_images.std(-1) + 1e-5)
    # b -> 1 value
    err = err.mean() / pixel_num
    return err

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


def compute_loss(predicted_images, images, structural_predicted_particles_ht, predicted_images_no_ctf, tracking_dict, loss_type="correlation"):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, side_shape**2), predicted images
    :param images: torch.tensor(batch_size, side_shape**2), images
    :param latent_mean:torch.tensor(batch_size, latent_dim), mean of the approximate latent distribution
    :param latent_std:torch.tensor(batch_size, latent_dim), std of the approximate latent distribution
    :param loss_weights: dict, containing the strength of losses for each loss
    :return:
    """
    if loss_type == "correlation":
        rmsd = calc_cor_loss(images, predicted_images)
    else:
        rmsd = compute_image_loss(images, predicted_images)

    loss_regularization = 0
    if structural_predicted_particles_ht is not None:
        rmsd_structural = compute_image_loss(predicted_images_no_ctf.flatten(start_dim=-2, end_dim=-1), structural_predicted_particles_ht.flatten(start_dim=-2, end_dim=-1))

    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["rmsd_structural"].append(rmsd_structural.detach().cpu().numpy())

    loss = rmsd
    return loss + 1*rmsd_structural