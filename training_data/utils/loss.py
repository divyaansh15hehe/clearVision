from torch import sum
import torch.nn.functional as F

def vae_loss_function(reconstructed, target, mu, logvar):
    recon_loss = F.l1_loss(reconstructed, target)
    kl_div = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div /= target.shape[0] * 3 * 32 * 32  # normalize for CIFAR-10
    return recon_loss + kl_div