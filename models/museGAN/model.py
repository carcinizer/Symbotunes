import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ..base import BaseModel


class museGAN(BaseModel):
    """museGAN-based music generation model"""

    def __init__(
        self,
        latent_dim: int,
        num_tracks: int,
        num_bars: int,
        num_steps_per_bar: int,
        num_pitches: int,
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        lr_decay: float = 0.97,
        lr_decay_start: int = 20,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.num_tracks = num_tracks
        self.num_bars = num_bars
        self.num_steps_per_bar = num_steps_per_bar
        self.num_pitches = num_pitches

        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.lr_decay = lr_decay
        self.lr_decay_start = lr_decay_start

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_tracks * num_bars * num_steps_per_bar * num_pitches),
            nn.Sigmoid(),
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(num_tracks * num_bars * num_steps_per_bar * num_pitches, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator."""
        generated = self.generator(z)
        return generated.view(-1, self.num_tracks, self.num_bars, self.num_steps_per_bar, self.num_pitches)

    def _discriminator_step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss."""
        real_scores = self.discriminator(real_data.view(real_data.size(0), -1))
        fake_scores = self.discriminator(fake_data.view(fake_data.size(0), -1))
        real_loss = nn.BCELoss()(real_scores, torch.ones_like(real_scores))
        fake_loss = nn.BCELoss()(fake_scores, torch.zeros_like(fake_scores))
        return (real_loss + fake_loss) / 2

    def _generator_step(self, fake_data: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        fake_scores = self.discriminator(fake_data.view(fake_data.size(0), -1))
        return nn.BCELoss()(fake_scores, torch.ones_like(fake_scores))

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_data = batch
        z = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
        fake_data = self(z)

        if optimizer_idx == 0:  # Generator step
            loss = self._generator_step(fake_data)
            self.log("train/generator_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss

        if optimizer_idx == 1:  # Discriminator step
            loss = self._discriminator_step(real_data, fake_data)
            self.log("train/discriminator_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss

    def configure_optimizers(self):
        generator_optimizer = Adam(self.generator.parameters(), lr=self.generator_lr)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.discriminator_lr)

        generator_scheduler = LambdaLR(
            generator_optimizer,
            lr_lambda=lambda epoch: (
                1 if epoch < self.lr_decay_start else self.lr_decay ** (epoch - self.lr_decay_start)
            ),
        )
        discriminator_scheduler = LambdaLR(
            discriminator_optimizer,
            lr_lambda=lambda epoch: (
                1 if epoch < self.lr_decay_start else self.lr_decay ** (epoch - self.lr_decay_start)
            ),
        )

        return [
            {"optimizer": generator_optimizer, "lr_scheduler": generator_scheduler},
            {"optimizer": discriminator_optimizer, "lr_scheduler": discriminator_scheduler},
        ]

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """Generate samples using the generator."""
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return self(z)