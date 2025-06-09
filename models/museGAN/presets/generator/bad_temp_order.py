"""This file defines the network architecture for the generator."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NORMALIZATION = 'batch_norm'
ACTIVATION = F.relu

class Generator(nn.Module):
    def __init__(self, n_tracks, latent_dim):
        super(Generator, self).__init__()
        self.n_tracks = n_tracks
        # Shared network
        self.conv1 = nn.ConvTranspose3d(latent_dim, 512, kernel_size=(4,1,1), stride=(4,1,1))
        self.bn1 = nn.BatchNorm3d(512)
        self.conv2 = nn.ConvTranspose3d(512,256,kernel_size=(1,3,3), stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.ConvTranspose3d(256,128,kernel_size=(1,4,3), stride=(1,4,2))
        self.bn3 = nn.BatchNorm3d(128)
        # Pitch-time private network
        self.pt_conv1 = nn.ModuleList([nn.ConvTranspose3d(128,32,kernel_size=(1,1,12),stride=(1,1,12)) for _ in range(n_tracks)])
        self.pt_bn1   = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.pt_conv2 = nn.ModuleList([nn.ConvTranspose3d(32,16,kernel_size=(1,4,1),stride=(1,4,1)) for _ in range(n_tracks)])
        self.pt_bn2   = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])
        # Time-pitch private network
        self.tp_conv1 = nn.ModuleList([nn.ConvTranspose3d(128,32,kernel_size=(1,4,1),stride=(1,4,1)) for _ in range(n_tracks)])
        self.tp_bn1   = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.tp_conv2 = nn.ModuleList([nn.ConvTranspose3d(32,16,kernel_size=(1,1,12),stride=(1,1,12)) for _ in range(n_tracks)])
        self.tp_bn2   = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])
        # Merged private network
        self.final_conv = nn.ConvTranspose3d(32,1,kernel_size=(1,1,1),stride=(1,1,1))
        self.final_bn   = nn.BatchNorm3d(1)

    def forward(self, x):
        # x: [batch, latent_dim]
        h = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Shared network
        h = ACTIVATION(self.bn1(self.conv1(h)))
        h = ACTIVATION(self.bn2(self.conv2(h)))
        h = ACTIVATION(self.bn3(self.conv3(h)))
        s1, s2 = [], []
        for i in range(self.n_tracks):
            y = ACTIVATION(self.pt_bn1[i](self.pt_conv1[i](h)))
            y = ACTIVATION(self.pt_bn2[i](self.pt_conv2[i](y)))
            s1.append(y)
            z = ACTIVATION(self.tp_bn1[i](self.tp_conv1[i](h)))
            z = ACTIVATION(self.tp_bn2[i](self.tp_conv2[i](z)))
            s2.append(z)
        # Merge and final output
        merged = [torch.cat((s1[i], s2[i]), dim=1) for i in range(self.n_tracks)]
        outs   = [self.final_bn(self.final_conv(m)) for m in merged]
        h      = torch.cat(outs, dim=1)
        return torch.tanh(h)
