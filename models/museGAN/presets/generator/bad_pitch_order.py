"""This file defines the network architecture for the generator."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, n_tracks, latent_dim):
        super(Generator, self).__init__()
        self.n_tracks = n_tracks
        self.latent_dim = latent_dim
        # Shared network
        self.shared = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 512, kernel_size=(4,1,1), stride=(4,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 256, kernel_size=(1,4,1), stride=(1,4,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(1,4,1), stride=(1,4,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(1,1,12), stride=(1,1,12)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        # Pitch-time private network
        self.pt_conv1 = nn.ModuleList([
            nn.ConvTranspose3d(64, 32, kernel_size=(1,1,7), stride=(1,1,7))
            for _ in range(n_tracks)
        ])
        self.pt_bn1 = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.pt_conv2 = nn.ModuleList([
            nn.ConvTranspose3d(32, 16, kernel_size=(1,3,1), stride=(1,3,1))
            for _ in range(n_tracks)
        ])
        self.pt_bn2 = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])
        # Time-pitch private network
        self.tp_conv1 = nn.ModuleList([
            nn.ConvTranspose3d(64, 32, kernel_size=(1,3,1), stride=(1,3,1))
            for _ in range(n_tracks)
        ])
        self.tp_bn1 = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.tp_conv2 = nn.ModuleList([
            nn.ConvTranspose3d(32, 16, kernel_size=(1,1,7), stride=(1,1,7))
            for _ in range(n_tracks)
        ])
        self.tp_bn2 = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])
        # Merged private network
        self.merge_conv = nn.ConvTranspose3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1))
        self.merge_bn = nn.BatchNorm3d(1)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, latent_dim)
        returns: generated output of shape (batch_size, n_tracks, depth, height, width)
        """
        batch_size = x.size(0)
        # reshape to (batch, channels, D, H, W)
        h = x.view(batch_size, self.latent_dim, 1, 1, 1)
        # shared layers
        h = self.shared(h)
        # private branches
        s1, s2 = [], []
        for i in range(self.n_tracks):
            t = self.pt_conv1[i](h)
            t = self.pt_bn1[i](t)
            t = F.relu(t)
            t = self.pt_conv2[i](t)
            t = self.pt_bn2[i](t)
            s1.append(F.relu(t))
            u = self.tp_conv1[i](h)
            u = self.tp_bn1[i](u)
            u = F.relu(u)
            u = self.tp_conv2[i](u)
            u = self.tp_bn2[i](u)
            s2.append(F.relu(u))
        # merge and combine tracks
        outputs = []
        for i in range(self.n_tracks):
            cat = torch.cat([s1[i], s2[i]], dim=1)
            m = self.merge_conv(cat)
            m = self.merge_bn(m)
            outputs.append(m)
        # concatenate along channel dimension: results in (batch, n_tracks, D, H, W)
        out = torch.cat(outputs, dim=1)
        return torch.tanh(out)
