"""This file defines the network architecture for the generator."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, n_tracks, latent_dim, condition_channels, condition_track_idx=None):
        super(Generator, self).__init__()
        self.n_tracks = n_tracks
        self.condition_track_idx = condition_track_idx

        # Encoder: pitch-time private
        self.pt_enc1 = nn.Conv3d(condition_channels, 16, kernel_size=(1,1,12), stride=(1,1,12))
        self.pt_bn1 = nn.BatchNorm3d(16)
        self.pt_enc2 = nn.Conv3d(16, 32, kernel_size=(1,3,1), stride=(1,3,1))
        self.pt_bn2 = nn.BatchNorm3d(32)

        # Encoder: time-pitch private
        self.tp_enc1 = nn.Conv3d(condition_channels, 16, kernel_size=(1,3,1), stride=(1,3,1))
        self.tp_bn1 = nn.BatchNorm3d(16)
        self.tp_enc2 = nn.Conv3d(16, 32, kernel_size=(1,1,12), stride=(1,1,12))
        self.tp_bn2 = nn.BatchNorm3d(32)

        # Encoder: shared
        self.enc_shared1 = nn.Conv3d(64, 64, kernel_size=(1,4,3), stride=(1,4,2))
        self.enc_bn3 = nn.BatchNorm3d(64)
        self.enc_shared2 = nn.Conv3d(64, 128, kernel_size=(1,4,3), stride=(1,4,3))
        self.enc_bn4 = nn.BatchNorm3d(128)
        self.enc_shared3 = nn.Conv3d(128, 256, kernel_size=(4,1,1), stride=(4,1,1))
        self.enc_bn5 = nn.BatchNorm3d(256)

        # Generator: shared decoder
        self.dec_shared1 = nn.ConvTranspose3d(latent_dim, 512, kernel_size=(4,1,1), stride=(4,1,1))
        self.dec_bn1 = nn.BatchNorm3d(512)
        self.dec_shared2 = nn.ConvTranspose3d(512, 256, kernel_size=(1,4,3), stride=(1,4,3))
        self.dec_bn2 = nn.BatchNorm3d(256)
        self.dec_shared3 = nn.ConvTranspose3d(256, 128, kernel_size=(1,4,3), stride=(1,4,2))
        self.dec_bn3 = nn.BatchNorm3d(128)

        # Generator: pitch-time private decoder
        self.pt_dec1 = nn.ModuleList([nn.ConvTranspose3d(128, 32, kernel_size=(1,1,12), stride=(1,1,12)) for _ in range(n_tracks)])
        self.pt_dec_bn1 = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.pt_dec2 = nn.ModuleList([nn.ConvTranspose3d(32, 16, kernel_size=(1,3,1), stride=(1,3,1)) for _ in range(n_tracks)])
        self.pt_dec_bn2 = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])

        # Generator: time-pitch private decoder
        self.tp_dec1 = nn.ModuleList([nn.ConvTranspose3d(128, 32, kernel_size=(1,3,1), stride=(1,3,1)) for _ in range(n_tracks)])
        self.tp_dec_bn1 = nn.ModuleList([nn.BatchNorm3d(32) for _ in range(n_tracks)])
        self.tp_dec2 = nn.ModuleList([nn.ConvTranspose3d(32, 16, kernel_size=(1,1,12), stride=(1,1,12)) for _ in range(n_tracks)])
        self.tp_dec_bn2 = nn.ModuleList([nn.BatchNorm3d(16) for _ in range(n_tracks)])

        # Merged private decoder
        self.merge_conv = nn.ConvTranspose3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1))
        self.merge_bn = nn.BatchNorm3d(1)

    def forward(self, tensor_in, condition_track):
        # Encoder
        c = condition_track  # shape: (batch, C, D, H, W)
        pt = F.relu(self.pt_bn1(self.pt_enc1(c)))
        pt = F.relu(self.pt_bn2(self.pt_enc2(pt)))
        tp = F.relu(self.tp_bn1(self.tp_enc1(c)))
        tp = F.relu(self.tp_bn2(self.tp_enc2(tp)))
        shared = torch.cat([tp, pt], dim=1)
        s1 = F.relu(self.enc_bn3(self.enc_shared1(shared)))
        s2 = F.relu(self.enc_bn4(self.enc_shared2(s1)))
        s3 = F.relu(self.enc_bn5(self.enc_shared3(s2)))

        # Decoder shared
        batch = tensor_in.size(0)
        h = tensor_in.view(batch, tensor_in.size(1), 1, 1, 1)
        h = F.relu(self.dec_bn1(self.dec_shared1(h)))
        h = F.relu(self.dec_bn2(self.dec_shared2(h)))
        h = F.relu(self.dec_bn3(self.dec_shared3(h)))

        # Private decoders
        outputs = []
        for i in range(self.n_tracks):
            p = F.relu(self.pt_dec_bn1[i](self.pt_dec1[i](h)))
            p = F.relu(self.pt_dec_bn2[i](self.pt_dec2[i](p)))
            u = F.relu(self.tp_dec_bn1[i](self.tp_dec1[i](h)))
            u = F.relu(self.tp_dec_bn2[i](self.tp_dec2[i](u)))
            cat = torch.cat([p, u], dim=1)
            m = self.merge_bn(self.merge_conv(cat))
            outputs.append(m)

        # Insert condition track at specified index
        if self.condition_track_idx is not None:
            outputs.insert(self.condition_track_idx, c)
        out = torch.cat(outputs, dim=1)
        return torch.tanh(out)
