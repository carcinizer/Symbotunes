"""This file defines common GAN losses."""
import torch
import torch.nn.functional as F

def get_adv_losses(discriminator_real_outputs, discriminator_fake_outputs,
                   kind):
    """Return the corresponding GAN losses for the generator and the
    discriminator."""
    if kind == 'classic':
        loss_fn = classic_gan_losses
    elif kind == 'nonsaturating':
        loss_fn = nonsaturating_gan_losses
    elif kind == 'wasserstein':
        loss_fn = wasserstein_gan_losses
    elif kind == 'hinge':
        loss_fn = hinge_gan_losses
    return loss_fn(discriminator_real_outputs, discriminator_fake_outputs)

def classic_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the classic GAN losses for the generator and the discriminator.

    (Generator)      log(1 - sigmoid(D(G(z))))
    (Discriminator)  - log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    # Create target tensors of ones and zeros
    ones = torch.ones_like(discriminator_real_outputs)
    zeros = torch.zeros_like(discriminator_fake_outputs)

    # Calculate BCE losses for discriminator
    discriminator_loss_real = F.binary_cross_entropy_with_logits(
        discriminator_real_outputs, ones)
    discriminator_loss_fake = F.binary_cross_entropy_with_logits(
        discriminator_fake_outputs, zeros)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = -discriminator_loss
    return generator_loss, discriminator_loss

def nonsaturating_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the non-saturating GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    # Create target tensors of ones and zeros
    ones_real = torch.ones_like(discriminator_real_outputs)
    zeros_fake = torch.zeros_like(discriminator_fake_outputs)
    ones_fake = torch.ones_like(discriminator_fake_outputs)

    # Calculate BCE losses for discriminator
    discriminator_loss_real = F.binary_cross_entropy_with_logits(
        discriminator_real_outputs, ones_real)
    discriminator_loss_fake = F.binary_cross_entropy_with_logits(
        discriminator_fake_outputs, zeros_fake)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake

    # Calculate BCE loss for generator (non-saturating version)
    generator_loss = F.binary_cross_entropy_with_logits(
        discriminator_fake_outputs, ones_fake)
    return generator_loss, discriminator_loss

def wasserstein_gan_losses(discriminator_real_outputs,
                           discriminator_fake_outputs):
    """Return the Wasserstein GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  D(G(z)) - D(x)
    """
    generator_loss = -torch.mean(discriminator_fake_outputs)
    discriminator_loss = -generator_loss - torch.mean(
        discriminator_real_outputs)
    return generator_loss, discriminator_loss

def hinge_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the Hinge GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = -torch.mean(discriminator_fake_outputs)
    discriminator_loss = (
        torch.mean(F.relu(1. - discriminator_real_outputs))
        + torch.mean(F.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

